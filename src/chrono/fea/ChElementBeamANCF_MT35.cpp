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
// ANCF beam element with 3 nodes.
// =============================================================================
//
// MT35 = MT32 with Analytical Jacobian updates to match the loop free internal force calculation
//  Mass Matrix = Constant, pre-calculated 27x27 matrix
//  Generalized Force due to gravity = Constant 27x1 Vector 
//     (assumption that gravity is constant too)
//  Generalized Internal Force Vector = Calculated in the typical paper way: 
//     e = 27x1 and S = 3x27
//     Inverse of the Element Jacobian (J_0xi) is generated from e0 every time
//     Math direct translation from papers
//     "Full Integration"-1 Number of GQ Integration Points (4x2x2)
//  Jacobian of the Generalized Internal Force Vector = Calculated by numeric 
//     differentiation
//
// =============================================================================

//// MIKE
//// Add function to manually set the initial configuration of the element (What update functions should this call?  Right now, if I change the material, I don't think the mass matrix gets updated
//// Should the Mass Matrix be stored as a sparse matrix? If so, make this change in a later revision
//// Check with Radu on function calls (e.g. calculated det_J_0xi).  If this can be improved, make this change in a later revision
//// Ask Radu about Heap vs. Stack calls with Matrices.
////
//// What should be done with ChVector<> ChElementBeamANCF_MT35::EvaluateBeamSectionStrains()... It doesn't seems to make sense for this element since there are no input arguments
////   In reality, there should be a function that returns the entire stress and strain tensors (which version(s)) is more of a question.
//// There is another block of base class functions that are commented out in the current Chrono implementation.  What needs to be done with these?
////
//// When is the right time to add new content, like the ability to apply a torque to an element (or different materials)?  Currently only applied forces appear to be supported
////
//// For ChElementBeamANCF_MT35::ComputeNF, Need to figure out the state update stuff and exactly what Jacobian is being asked for
////   Can the 1D versions just call the 3D version to eliminate code duplication?
////
//// What is "EvaluateSectionVelNorm(double U, ChVector<>& Result)"?  Is it even needed?
////
//// Figure out what is missing to enable irrlicht animations (missing "beam section" functions...How does this affect the entire interface for the element?)


//// RADU
//// A lot more to do here...
//// - more use of Eigen expressions
//// - remove unecessary initializations to zero

#include "chrono/core/ChQuadrature.h"
#include "chrono/physics/ChSystem.h"
#include "chrono/fea/ChElementBeamANCF_MT35.h"
#include <cmath>
#include <Eigen/Dense>

namespace chrono {
namespace fea {


// ------------------------------------------------------------------------------
// Constructor
// ------------------------------------------------------------------------------

ChElementBeamANCF_MT35::ChElementBeamANCF_MT35() : m_gravity_on(false), m_thicknessY(0), m_thicknessZ(0), m_lenX(0), m_Alpha(0), m_damping_enabled(false){
    m_nodes.resize(3);

    m_F_Transpose_CombinedBlock_col_ordered.setZero();
    m_F_Transpose_CombinedBlockDamping_col_ordered.setZero();
    m_SPK2_0_D0_Block.setZero();
    m_SPK2_1_D0_Block.setZero();
    m_SPK2_2_D0_Block.setZero();
    m_SPK2_3_D0_Block.setZero();
    m_SPK2_4_D0_Block.setZero();
    m_SPK2_5_D0_Block.setZero();
    m_Sdiag_0_Dv_Block.setZero();
    m_Sdiag_1_Dv_Block.setZero();
    m_Sdiag_2_Dv_Block.setZero();
}

// ------------------------------------------------------------------------------
// Set element nodes
// ------------------------------------------------------------------------------

void ChElementBeamANCF_MT35::SetNodes(std::shared_ptr<ChNodeFEAxyzDD> nodeA,
                                 std::shared_ptr<ChNodeFEAxyzDD> nodeB,
                                 std::shared_ptr<ChNodeFEAxyzDD> nodeC) {
    assert(nodeA);
    assert(nodeB);
    assert(nodeC);

    m_nodes[0] = nodeA;
    m_nodes[1] = nodeB;
    m_nodes[2] = nodeC;

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

    Kmatr.SetVariables(mvars);

    // Initial positions and slopes of the element nodes
    // These values define the reference configuration of the element
    CalcCoordMatrix(m_e0_bar);
}

// -----------------------------------------------------------------------------
// Interface to ChElementBase base class
// -----------------------------------------------------------------------------

// Initial element setup.
void ChElementBeamANCF_MT35::SetupInitial(ChSystem* system) {
    // Compute mass matrix and gravitational forces and store them since they are constants
    ComputeMassMatrixAndGravityForce(system->Get_G_acc());
    PrecomputeInternalForceMatricesWeights();
}

// State update.
void ChElementBeamANCF_MT35::Update() {
    ChElementGeneric::Update();
}

// Fill the D vector with the current field values at the element nodes.
void ChElementBeamANCF_MT35::GetStateBlock(ChVectorDynamic<>& mD) {
    mD.segment(0, 3) = m_nodes[0]->GetPos().eigen();
    mD.segment(3, 3) = m_nodes[0]->GetD().eigen();
    mD.segment(6, 3) = m_nodes[0]->GetDD().eigen();
    mD.segment(9, 3) = m_nodes[1]->GetPos().eigen();
    mD.segment(12, 3) = m_nodes[1]->GetD().eigen();
    mD.segment(15, 3) = m_nodes[1]->GetDD().eigen();
    mD.segment(18, 3) = m_nodes[2]->GetPos().eigen();
    mD.segment(21, 3) = m_nodes[2]->GetD().eigen();
    mD.segment(24, 3) = m_nodes[2]->GetDD().eigen();
}

// Calculate the global matrix H as a linear combination of K, R, and M:
//   H = Mfactor * [M] + Kfactor * [K] + Rfactor * [R]
void ChElementBeamANCF_MT35::ComputeKRMmatricesGlobal(ChMatrixRef H, double Kfactor, double Rfactor, double Mfactor) {
    assert((H.rows() == 27) && (H.cols() == 27));

    ////Use H to accumulate the Dense part of the Jacobian, so set it to all zeros
    ////H.setZero();

#if true

    if (m_damping_enabled) {        //If linear Kelvin-Voigt viscoelastic material model is enabled 
        ComputeInternalJacobianDamping(H, Kfactor, Rfactor, Mfactor);
    }
    else {
        ComputeInternalJacobianNoDamping(H, Kfactor, Rfactor, Mfactor);
    }

    ////Start the accumulation of the Compact part of the Jacobian with the scaled
    ////compact mass matrix since this needs to be added anyways and this
    ////eliminates the need to initialize this accumulator with all zeros.
    ////ChMatrixNM<double, 9, 9> Jacobian_CompactPart = Mfactor * m_MassMatrix;
    ////ChMatrixNM<double, 27, 27> Jacobian_DensePart;
    ////ChMatrixNMc<double, 9, 3> e_bar;
    ////ChMatrixNMc<double, 9, 3> e_bar_dot;
    //ChVectorN<double, 6> D0 = GetMaterial()->Get_D0();
    //ChMatrix33<double> Dv = GetMaterial()->Get_Dv();

    //Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor, 9, 9> Jacobian_CompactPart;
    //Jacobian_CompactPart.resize(9, 9);

    ////CalcCoordMatrix(e_bar);
    ////CalcCoordDerivMatrix(e_bar_dot);
    ////Jacobian_DensePart.setZero();
    //////Jacobian_CompactPart.setZero();

    //////Calculate the portion of the Selective Reduced Integration that does account for the Poisson effect
    ////for (unsigned int index = 0; index < m_GQWeight_det_J_0xi_D0.size(); index++) {
    ////    ComputeInternalJacobianSingleGQPnt(Jacobian_DensePart, Jacobian_CompactPart, -Kfactor, -Rfactor, index, D0, e_bar, e_bar_dot);
    ////}

    //////Calculate the portion of the Selective Reduced Integration that account for the Poisson effect, but only on the beam axis
    ////if (GetStrainFormulation() == ChElementBeamANCF_MT35::StrainFormulation::CMPoisson) {
    ////    for (unsigned int index = 0; index < m_GQWeight_det_J_0xi_Dv.size(); index++) {
    ////        ComputeInternalJacobianSingleGQPnt(Jacobian_DensePart, Jacobian_CompactPart, -Kfactor, -Rfactor, index, Dv, e_bar, e_bar_dot);
    ////    }
    ////}

    ////H = Jacobian_DensePart;

    //////Inflate the Mass Matrix since it is stored in compact form.
    //////In MATLAB notation:	 
    //////H(1:3:end,1:3:end) = H(1:3:end,1:3:end) + Jacobian_CompactPart;
    //////H(2:3:end,2:3:end) = H(2:3:end,2:3:end) + Jacobian_CompactPart;
    //////H(3:3:end,3:3:end) = H(3:3:end,3:3:end) + Jacobian_CompactPart;
    ////for (unsigned int i = 0; i < 9; i++) {
    ////    for (unsigned int j = 0; j < 9; j++) {
    ////        H(3 * i, 3 * j) += Jacobian_CompactPart(i, j);
    ////        H(3 * i + 1, 3 * j + 1) += Jacobian_CompactPart(i, j);
    ////        H(3 * i + 2, 3 * j + 2) += Jacobian_CompactPart(i, j);
    ////    }
    ////}

    ////===========================================================================================

    ////SD_mod(:, (1:16) + 0 * 16) = repmat(SPK2_0_D0_Block',NumShapeFunctions,1).*SD_reordered(:,(1:16)+0*16)+...
    ////repmat(SPK2_5_D0_Block',NumShapeFunctions,1).*SD_reordered(:,(1:16)+1*16)+...
    ////repmat(SPK2_4_D0_Block',NumShapeFunctions,1).*SD_reordered(:,(1:16)+2*16);

    ////SD_mod(:, (1:16) + 1 * 16) = repmat(SPK2_5_D0_Block',NumShapeFunctions,1).*SD_reordered(:,(1:16)+0*16)+...
    ////repmat(SPK2_1_D0_Block',NumShapeFunctions,1).*SD_reordered(:,(1:16)+1*16)+...
    ////repmat(SPK2_3_D0_Block',NumShapeFunctions,1).*SD_reordered(:,(1:16)+2*16);

    ////SD_mod(:, (1:16) + 2 * 16) = repmat(SPK2_4_D0_Block',NumShapeFunctions,1).*SD_reordered(:,(1:16)+0*16)+...
    ////repmat(SPK2_3_D0_Block',NumShapeFunctions,1).*SD_reordered(:,(1:16)+1*16)+...
    ////repmat(SPK2_2_D0_Block',NumShapeFunctions,1).*SD_reordered(:,(1:16)+2*16);

    ////SD_mod(:, (1:4) + 3 * 16 + 0) = repmat(Sdiag_0_Dv_Block',NumShapeFunctions,1).*SD_reordered(:,(1:4)+3*16+0);
    ////SD_mod(:, (1:4) + 3 * 16 + 4) = repmat(Sdiag_1_Dv_Block',NumShapeFunctions,1).*SD_reordered(:,(1:4)+3*16+4);
    ////SD_mod(:, (1:4) + 3 * 16 + 8) = repmat(Sdiag_2_Dv_Block',NumShapeFunctions,1).*SD_reordered(:,(1:4)+3*16+8);

    ////JacCompactVectorized = SD_reordered * SD_mod';

    ////ChMatrixNMc<double, 9, 60> S_scaled_SD_precompute_col_ordered;
    //Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor, 9, 60> S_scaled_SD_precompute_col_ordered;
    //S_scaled_SD_precompute_col_ordered.resize(9, 60);

    //for (auto i = 0; i < 9; i++) {
    //    S_scaled_SD_precompute_col_ordered.block<1, 16>(i, 0) = m_SPK2_0_D0_Block.transpose().cwiseProduct(m_SD_precompute_col_ordered.block<1, 16>(i, 0));
    //    S_scaled_SD_precompute_col_ordered.block<1, 16>(i, 0) += m_SPK2_5_D0_Block.transpose().cwiseProduct(m_SD_precompute_col_ordered.block<1, 16>(i, 16));
    //    S_scaled_SD_precompute_col_ordered.block<1, 16>(i, 0) += m_SPK2_4_D0_Block.transpose().cwiseProduct(m_SD_precompute_col_ordered.block<1, 16>(i, 32));

    //    S_scaled_SD_precompute_col_ordered.block<1, 16>(i, 16) = m_SPK2_5_D0_Block.transpose().cwiseProduct(m_SD_precompute_col_ordered.block<1, 16>(i, 0));
    //    S_scaled_SD_precompute_col_ordered.block<1, 16>(i, 16) += m_SPK2_1_D0_Block.transpose().cwiseProduct(m_SD_precompute_col_ordered.block<1, 16>(i, 16));
    //    S_scaled_SD_precompute_col_ordered.block<1, 16>(i, 16) += m_SPK2_3_D0_Block.transpose().cwiseProduct(m_SD_precompute_col_ordered.block<1, 16>(i, 32));

    //    S_scaled_SD_precompute_col_ordered.block<1, 16>(i, 32) = m_SPK2_4_D0_Block.transpose().cwiseProduct(m_SD_precompute_col_ordered.block<1, 16>(i, 0));
    //    S_scaled_SD_precompute_col_ordered.block<1, 16>(i, 32) += m_SPK2_3_D0_Block.transpose().cwiseProduct(m_SD_precompute_col_ordered.block<1, 16>(i, 16));
    //    S_scaled_SD_precompute_col_ordered.block<1, 16>(i, 32) += m_SPK2_2_D0_Block.transpose().cwiseProduct(m_SD_precompute_col_ordered.block<1, 16>(i, 32));

    //    S_scaled_SD_precompute_col_ordered.block<1, 4>(i, 48) = m_Sdiag_0_Dv_Block.transpose().cwiseProduct(m_SD_precompute_col_ordered.block<1, 4>(i, 48));
    //    S_scaled_SD_precompute_col_ordered.block<1, 4>(i, 52) = m_Sdiag_0_Dv_Block.transpose().cwiseProduct(m_SD_precompute_col_ordered.block<1, 4>(i, 52));
    //    S_scaled_SD_precompute_col_ordered.block<1, 4>(i, 56) = m_Sdiag_0_Dv_Block.transpose().cwiseProduct(m_SD_precompute_col_ordered.block<1, 4>(i, 56));
    //}

    ////Jacobian_CompactPart += m_SD_precompute_col_ordered * S_scaled_SD_precompute_col_ordered.transpose();
    //Jacobian_CompactPart = Mfactor * m_MassMatrix - Kfactor * m_SD_precompute_col_ordered * S_scaled_SD_precompute_col_ordered.transpose();
    ////ChMatrixNM<double, 9, 9> Jacobian_CompactPart2 = -Kfactor*m_SD_precompute_col_ordered * S_scaled_SD_precompute_col_ordered.transpose();


    //////ChunkA1 = SD_reordered(:, (GQPnts)+0 * 16).*repmat(Fcol0_Transpose_D0_Block(GQPnts, 1)',NumShapeFunctions,1);
    //////ChunkA2 = SD_reordered(:, (GQPnts)+0 * 16).*repmat(Fcol0_Transpose_D0_Block(GQPnts, 2)',NumShapeFunctions,1);
    //////ChunkA3 = SD_reordered(:, (GQPnts)+0 * 16).*repmat(Fcol0_Transpose_D0_Block(GQPnts, 3)',NumShapeFunctions,1);
    //////ChunkA4 = SD_reordered(:, (GQPnts)+1 * 16).*repmat(Fcol0_Transpose_D0_Block(GQPnts, 1)',NumShapeFunctions,1);
    //////ChunkA5 = SD_reordered(:, (GQPnts)+1 * 16).*repmat(Fcol0_Transpose_D0_Block(GQPnts, 2)',NumShapeFunctions,1);
    //////ChunkA6 = SD_reordered(:, (GQPnts)+1 * 16).*repmat(Fcol0_Transpose_D0_Block(GQPnts, 3)',NumShapeFunctions,1);
    //////ChunkA7 = SD_reordered(:, (GQPnts)+2 * 16).*repmat(Fcol0_Transpose_D0_Block(GQPnts, 1)',NumShapeFunctions,1);
    //////ChunkA8 = SD_reordered(:, (GQPnts)+2 * 16).*repmat(Fcol0_Transpose_D0_Block(GQPnts, 2)',NumShapeFunctions,1);
    //////ChunkA9 = SD_reordered(:, (GQPnts)+2 * 16).*repmat(Fcol0_Transpose_D0_Block(GQPnts, 3)',NumShapeFunctions,1);

    //////ChunkB1 = SD_reordered(:, (GQPnts)+0 * 16).*repmat(Fcol1_Transpose_D0_Block(GQPnts, 1)',NumShapeFunctions,1);
    //////ChunkB2 = SD_reordered(:, (GQPnts)+0 * 16).*repmat(Fcol1_Transpose_D0_Block(GQPnts, 2)',NumShapeFunctions,1);
    //////ChunkB3 = SD_reordered(:, (GQPnts)+0 * 16).*repmat(Fcol1_Transpose_D0_Block(GQPnts, 3)',NumShapeFunctions,1);
    //////ChunkB4 = SD_reordered(:, (GQPnts)+1 * 16).*repmat(Fcol1_Transpose_D0_Block(GQPnts, 1)',NumShapeFunctions,1);
    //////ChunkB5 = SD_reordered(:, (GQPnts)+1 * 16).*repmat(Fcol1_Transpose_D0_Block(GQPnts, 2)',NumShapeFunctions,1);
    //////ChunkB6 = SD_reordered(:, (GQPnts)+1 * 16).*repmat(Fcol1_Transpose_D0_Block(GQPnts, 3)',NumShapeFunctions,1);
    //////ChunkB7 = SD_reordered(:, (GQPnts)+2 * 16).*repmat(Fcol1_Transpose_D0_Block(GQPnts, 1)',NumShapeFunctions,1);
    //////ChunkB8 = SD_reordered(:, (GQPnts)+2 * 16).*repmat(Fcol1_Transpose_D0_Block(GQPnts, 2)',NumShapeFunctions,1);
    //////ChunkB9 = SD_reordered(:, (GQPnts)+2 * 16).*repmat(Fcol1_Transpose_D0_Block(GQPnts, 3)',NumShapeFunctions,1);

    //////ChunkC1 = SD_reordered(:, (GQPnts)+0 * 16).*repmat(Fcol2_Transpose_D0_Block(GQPnts, 1)',NumShapeFunctions,1);
    //////ChunkC2 = SD_reordered(:, (GQPnts)+0 * 16).*repmat(Fcol2_Transpose_D0_Block(GQPnts, 2)',NumShapeFunctions,1);
    //////ChunkC3 = SD_reordered(:, (GQPnts)+0 * 16).*repmat(Fcol2_Transpose_D0_Block(GQPnts, 3)',NumShapeFunctions,1);
    //////ChunkC4 = SD_reordered(:, (GQPnts)+1 * 16).*repmat(Fcol2_Transpose_D0_Block(GQPnts, 1)',NumShapeFunctions,1);
    //////ChunkC5 = SD_reordered(:, (GQPnts)+1 * 16).*repmat(Fcol2_Transpose_D0_Block(GQPnts, 2)',NumShapeFunctions,1);
    //////ChunkC6 = SD_reordered(:, (GQPnts)+1 * 16).*repmat(Fcol2_Transpose_D0_Block(GQPnts, 3)',NumShapeFunctions,1);
    //////ChunkC7 = SD_reordered(:, (GQPnts)+2 * 16).*repmat(Fcol2_Transpose_D0_Block(GQPnts, 1)',NumShapeFunctions,1);
    //////ChunkC8 = SD_reordered(:, (GQPnts)+2 * 16).*repmat(Fcol2_Transpose_D0_Block(GQPnts, 2)',NumShapeFunctions,1);
    //////ChunkC9 = SD_reordered(:, (GQPnts)+2 * 16).*repmat(Fcol2_Transpose_D0_Block(GQPnts, 3)',NumShapeFunctions,1);

    //////Fcol0_Transpose_D0_Block.array().colwise()*SPK2_4_D0_Block.array()

    //Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor, 9, 16> ChunkA1;
    //ChunkA1.resize(9, 16);
    //Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor, 9, 16> ChunkA2;
    //ChunkA2.resize(9, 16);
    //Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor, 9, 16> ChunkA3;
    //ChunkA3.resize(9, 16);
    //Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor, 9, 16> ChunkA4;
    //ChunkA4.resize(9, 16);
    //Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor, 9, 16> ChunkA5;
    //ChunkA5.resize(9, 16);
    //Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor, 9, 16> ChunkA6;
    //ChunkA6.resize(9, 16);
    //Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor, 9, 16> ChunkA7;
    //ChunkA7.resize(9, 16);
    //Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor, 9, 16> ChunkA8;
    //ChunkA8.resize(9, 16);
    //Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor, 9, 16> ChunkA9;
    //ChunkA9.resize(9, 16);

    //ChunkA1 = m_SD_precompute_col_ordered.block<9, 16>(0, 0).array().rowwise()*m_F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(0, 0).array().transpose();
    //ChunkA2 = m_SD_precompute_col_ordered.block<9, 16>(0, 0).array().rowwise()*m_F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(0, 1).array().transpose();
    //ChunkA3 = m_SD_precompute_col_ordered.block<9, 16>(0, 0).array().rowwise()*m_F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(0, 2).array().transpose();
    //ChunkA4 = m_SD_precompute_col_ordered.block<9, 16>(0, 16).array().rowwise()*m_F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(0, 0).array().transpose();
    //ChunkA5 = m_SD_precompute_col_ordered.block<9, 16>(0, 16).array().rowwise()*m_F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(0, 1).array().transpose();
    //ChunkA6 = m_SD_precompute_col_ordered.block<9, 16>(0, 16).array().rowwise()*m_F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(0, 2).array().transpose();
    //ChunkA7 = m_SD_precompute_col_ordered.block<9, 16>(0, 32).array().rowwise()*m_F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(0, 0).array().transpose();
    //ChunkA8 = m_SD_precompute_col_ordered.block<9, 16>(0, 32).array().rowwise()*m_F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(0, 1).array().transpose();
    //ChunkA9 = m_SD_precompute_col_ordered.block<9, 16>(0, 32).array().rowwise()*m_F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(0, 2).array().transpose();

    //
    ////ChMatrixNMc<double, 9, 16> ChunkA1 = m_SD_precompute_col_ordered.block<9, 16>(0, 0).array().rowwise()*m_F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(0, 0).array().transpose();
    ////ChMatrixNMc<double, 9, 16> ChunkA2 = m_SD_precompute_col_ordered.block<9, 16>(0, 0).array().rowwise()*m_F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(0, 1).array().transpose();
    ////ChMatrixNMc<double, 9, 16> ChunkA3 = m_SD_precompute_col_ordered.block<9, 16>(0, 0).array().rowwise()*m_F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(0, 2).array().transpose();
    ////ChMatrixNMc<double, 9, 16> ChunkA4 = m_SD_precompute_col_ordered.block<9, 16>(0, 16).array().rowwise()*m_F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(0, 0).array().transpose();
    ////ChMatrixNMc<double, 9, 16> ChunkA5 = m_SD_precompute_col_ordered.block<9, 16>(0, 16).array().rowwise()*m_F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(0, 1).array().transpose();
    ////ChMatrixNMc<double, 9, 16> ChunkA6 = m_SD_precompute_col_ordered.block<9, 16>(0, 16).array().rowwise()*m_F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(0, 2).array().transpose();
    ////ChMatrixNMc<double, 9, 16> ChunkA7 = m_SD_precompute_col_ordered.block<9, 16>(0, 32).array().rowwise()*m_F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(0, 0).array().transpose();
    ////ChMatrixNMc<double, 9, 16> ChunkA8 = m_SD_precompute_col_ordered.block<9, 16>(0, 32).array().rowwise()*m_F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(0, 1).array().transpose();
    ////ChMatrixNMc<double, 9, 16> ChunkA9 = m_SD_precompute_col_ordered.block<9, 16>(0, 32).array().rowwise()*m_F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(0, 2).array().transpose();

    //Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor, 9, 16> ChunkB1;
    //ChunkB1.resize(9, 16);
    //Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor, 9, 16> ChunkB2;
    //ChunkB2.resize(9, 16);
    //Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor, 9, 16> ChunkB3;
    //ChunkB3.resize(9, 16);
    //Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor, 9, 16> ChunkB4;
    //ChunkB4.resize(9, 16);
    //Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor, 9, 16> ChunkB5;
    //ChunkB5.resize(9, 16);
    //Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor, 9, 16> ChunkB6;
    //ChunkB6.resize(9, 16);
    //Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor, 9, 16> ChunkB7;
    //ChunkB7.resize(9, 16);
    //Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor, 9, 16> ChunkB8;
    //ChunkB8.resize(9, 16);
    //Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor, 9, 16> ChunkB9;
    //ChunkB9.resize(9, 16);

    //ChunkB1 = m_SD_precompute_col_ordered.block<9, 16>(0, 0).array().rowwise()*m_F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(16, 0).array().transpose();
    //ChunkB2 = m_SD_precompute_col_ordered.block<9, 16>(0, 0).array().rowwise()*m_F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(16, 1).array().transpose();
    //ChunkB3 = m_SD_precompute_col_ordered.block<9, 16>(0, 0).array().rowwise()*m_F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(16, 2).array().transpose();
    //ChunkB4 = m_SD_precompute_col_ordered.block<9, 16>(0, 16).array().rowwise()*m_F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(16, 0).array().transpose();
    //ChunkB5 = m_SD_precompute_col_ordered.block<9, 16>(0, 16).array().rowwise()*m_F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(16, 1).array().transpose();
    //ChunkB6 = m_SD_precompute_col_ordered.block<9, 16>(0, 16).array().rowwise()*m_F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(16, 2).array().transpose();
    //ChunkB7 = m_SD_precompute_col_ordered.block<9, 16>(0, 32).array().rowwise()*m_F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(16, 0).array().transpose();
    //ChunkB8 = m_SD_precompute_col_ordered.block<9, 16>(0, 32).array().rowwise()*m_F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(16, 1).array().transpose();
    //ChunkB9 = m_SD_precompute_col_ordered.block<9, 16>(0, 32).array().rowwise()*m_F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(16, 2).array().transpose();

    ////ChMatrixNMc<double, 9, 16> ChunkB1 = m_SD_precompute_col_ordered.block<9, 16>(0, 0).array().rowwise()*m_F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(16, 0).array().transpose();
    ////ChMatrixNMc<double, 9, 16> ChunkB2 = m_SD_precompute_col_ordered.block<9, 16>(0, 0).array().rowwise()*m_F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(16, 1).array().transpose();
    ////ChMatrixNMc<double, 9, 16> ChunkB3 = m_SD_precompute_col_ordered.block<9, 16>(0, 0).array().rowwise()*m_F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(16, 2).array().transpose();
    ////ChMatrixNMc<double, 9, 16> ChunkB4 = m_SD_precompute_col_ordered.block<9, 16>(0, 16).array().rowwise()*m_F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(16, 0).array().transpose();
    ////ChMatrixNMc<double, 9, 16> ChunkB5 = m_SD_precompute_col_ordered.block<9, 16>(0, 16).array().rowwise()*m_F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(16, 1).array().transpose();
    ////ChMatrixNMc<double, 9, 16> ChunkB6 = m_SD_precompute_col_ordered.block<9, 16>(0, 16).array().rowwise()*m_F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(16, 2).array().transpose();
    ////ChMatrixNMc<double, 9, 16> ChunkB7 = m_SD_precompute_col_ordered.block<9, 16>(0, 32).array().rowwise()*m_F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(16, 0).array().transpose();
    ////ChMatrixNMc<double, 9, 16> ChunkB8 = m_SD_precompute_col_ordered.block<9, 16>(0, 32).array().rowwise()*m_F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(16, 1).array().transpose();
    ////ChMatrixNMc<double, 9, 16> ChunkB9 = m_SD_precompute_col_ordered.block<9, 16>(0, 32).array().rowwise()*m_F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(16, 2).array().transpose();

    //Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor, 9, 16> ChunkC1;
    //ChunkC1.resize(9, 16);
    //Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor, 9, 16> ChunkC2;
    //ChunkC2.resize(9, 16);
    //Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor, 9, 16> ChunkC3;
    //ChunkC3.resize(9, 16);
    //Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor, 9, 16> ChunkC4;
    //ChunkC4.resize(9, 16);
    //Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor, 9, 16> ChunkC5;
    //ChunkC5.resize(9, 16);
    //Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor, 9, 16> ChunkC6;
    //ChunkC6.resize(9, 16);
    //Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor, 9, 16> ChunkC7;
    //ChunkC7.resize(9, 16);
    //Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor, 9, 16> ChunkC8;
    //ChunkC8.resize(9, 16);
    //Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor, 9, 16> ChunkC9;
    //ChunkC9.resize(9, 16);

    //ChunkC1 = m_SD_precompute_col_ordered.block<9, 16>(0, 0).array().rowwise()*m_F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(32, 0).array().transpose();
    //ChunkC2 = m_SD_precompute_col_ordered.block<9, 16>(0, 0).array().rowwise()*m_F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(32, 1).array().transpose();
    //ChunkC3 = m_SD_precompute_col_ordered.block<9, 16>(0, 0).array().rowwise()*m_F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(32, 2).array().transpose();
    //ChunkC4 = m_SD_precompute_col_ordered.block<9, 16>(0, 16).array().rowwise()*m_F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(32, 0).array().transpose();
    //ChunkC5 = m_SD_precompute_col_ordered.block<9, 16>(0, 16).array().rowwise()*m_F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(32, 1).array().transpose();
    //ChunkC6 = m_SD_precompute_col_ordered.block<9, 16>(0, 16).array().rowwise()*m_F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(32, 2).array().transpose();
    //ChunkC7 = m_SD_precompute_col_ordered.block<9, 16>(0, 32).array().rowwise()*m_F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(32, 0).array().transpose();
    //ChunkC8 = m_SD_precompute_col_ordered.block<9, 16>(0, 32).array().rowwise()*m_F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(32, 1).array().transpose();
    //ChunkC9 = m_SD_precompute_col_ordered.block<9, 16>(0, 32).array().rowwise()*m_F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(32, 2).array().transpose();

    ////ChMatrixNMc<double, 9, 16> ChunkC1 = m_SD_precompute_col_ordered.block<9, 16>(0, 0).array().rowwise()*m_F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(32, 0).array().transpose();
    ////ChMatrixNMc<double, 9, 16> ChunkC2 = m_SD_precompute_col_ordered.block<9, 16>(0, 0).array().rowwise()*m_F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(32, 1).array().transpose();
    ////ChMatrixNMc<double, 9, 16> ChunkC3 = m_SD_precompute_col_ordered.block<9, 16>(0, 0).array().rowwise()*m_F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(32, 2).array().transpose();
    ////ChMatrixNMc<double, 9, 16> ChunkC4 = m_SD_precompute_col_ordered.block<9, 16>(0, 16).array().rowwise()*m_F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(32, 0).array().transpose();
    ////ChMatrixNMc<double, 9, 16> ChunkC5 = m_SD_precompute_col_ordered.block<9, 16>(0, 16).array().rowwise()*m_F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(32, 1).array().transpose();
    ////ChMatrixNMc<double, 9, 16> ChunkC6 = m_SD_precompute_col_ordered.block<9, 16>(0, 16).array().rowwise()*m_F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(32, 2).array().transpose();
    ////ChMatrixNMc<double, 9, 16> ChunkC7 = m_SD_precompute_col_ordered.block<9, 16>(0, 32).array().rowwise()*m_F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(32, 0).array().transpose();
    ////ChMatrixNMc<double, 9, 16> ChunkC8 = m_SD_precompute_col_ordered.block<9, 16>(0, 32).array().rowwise()*m_F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(32, 1).array().transpose();
    ////ChMatrixNMc<double, 9, 16> ChunkC9 = m_SD_precompute_col_ordered.block<9, 16>(0, 32).array().rowwise()*m_F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(32, 2).array().transpose();


    ////ChunkDvA1 = SD_reordered(:, (GQPnts)+2 * 16 + 0).*repmat(Fcol0_Transpose_Dv_Block(GQPnts - 16, 1)',NumShapeFunctions,1);
    ////ChunkDvA2 = SD_reordered(:, (GQPnts)+2 * 16 + 0).*repmat(Fcol0_Transpose_Dv_Block(GQPnts - 16, 2)',NumShapeFunctions,1);
    ////ChunkDvA3 = SD_reordered(:, (GQPnts)+2 * 16 + 0).*repmat(Fcol0_Transpose_Dv_Block(GQPnts - 16, 3)',NumShapeFunctions,1);

    ////ChunkDvB4 = SD_reordered(:, (GQPnts)+2 * 16 + 4).*repmat(Fcol1_Transpose_Dv_Block(GQPnts - 16, 1)',NumShapeFunctions,1);
    ////ChunkDvB5 = SD_reordered(:, (GQPnts)+2 * 16 + 4).*repmat(Fcol1_Transpose_Dv_Block(GQPnts - 16, 2)',NumShapeFunctions,1);
    ////ChunkDvB6 = SD_reordered(:, (GQPnts)+2 * 16 + 4).*repmat(Fcol1_Transpose_Dv_Block(GQPnts - 16, 3)',NumShapeFunctions,1);

    ////ChunkDvC7 = SD_reordered(:, (GQPnts)+2 * 16 + 8).*repmat(Fcol2_Transpose_Dv_Block(GQPnts - 16, 1)',NumShapeFunctions,1);
    ////ChunkDvC8 = SD_reordered(:, (GQPnts)+2 * 16 + 8).*repmat(Fcol2_Transpose_Dv_Block(GQPnts - 16, 2)',NumShapeFunctions,1);
    ////ChunkDvC9 = SD_reordered(:, (GQPnts)+2 * 16 + 8).*repmat(Fcol2_Transpose_Dv_Block(GQPnts - 16, 3)',NumShapeFunctions,1);

    //Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor, 9, 4> ChunkDvA1;
    //ChunkDvA1.resize(9, 4);
    //Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor, 9, 4> ChunkDvA2;
    //ChunkDvA2.resize(9, 4);
    //Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor, 9, 4> ChunkDvA3;
    //ChunkDvA3.resize(9, 4);

    //Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor, 9, 4> ChunkDvB4;
    //ChunkDvB4.resize(9, 4);
    //Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor, 9, 4> ChunkDvB5;
    //ChunkDvB5.resize(9, 4);
    //Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor, 9, 4> ChunkDvB6;
    //ChunkDvB6.resize(9, 4);

    //Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor, 9, 4> ChunkDvC7;
    //ChunkDvC7.resize(9, 4);
    //Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor, 9, 4> ChunkDvC8;
    //ChunkDvC8.resize(9, 4);
    //Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor, 9, 4> ChunkDvC9;
    //ChunkDvC9.resize(9, 4);

    ////ChMatrixNMc<double, 9, 4> ChunkDvA1 = m_SD_precompute_col_ordered.block<9, 4>(0, 48).array().rowwise()*m_F_Transpose_CombinedBlockDamping_col_ordered.block<4, 1>(48, 0).array().transpose();
    ////ChMatrixNMc<double, 9, 4> ChunkDvA2 = m_SD_precompute_col_ordered.block<9, 4>(0, 48).array().rowwise()*m_F_Transpose_CombinedBlockDamping_col_ordered.block<4, 1>(48, 1).array().transpose();
    ////ChMatrixNMc<double, 9, 4> ChunkDvA3 = m_SD_precompute_col_ordered.block<9, 4>(0, 48).array().rowwise()*m_F_Transpose_CombinedBlockDamping_col_ordered.block<4, 1>(48, 2).array().transpose();

    ////ChMatrixNMc<double, 9, 4> ChunkDvB4 = m_SD_precompute_col_ordered.block<9, 4>(0, 52).array().rowwise()*m_F_Transpose_CombinedBlockDamping_col_ordered.block<4, 1>(52, 0).array().transpose();
    ////ChMatrixNMc<double, 9, 4> ChunkDvB5 = m_SD_precompute_col_ordered.block<9, 4>(0, 52).array().rowwise()*m_F_Transpose_CombinedBlockDamping_col_ordered.block<4, 1>(52, 1).array().transpose();
    ////ChMatrixNMc<double, 9, 4> ChunkDvB6 = m_SD_precompute_col_ordered.block<9, 4>(0, 52).array().rowwise()*m_F_Transpose_CombinedBlockDamping_col_ordered.block<4, 1>(52, 2).array().transpose();

    ////ChMatrixNMc<double, 9, 4> ChunkDvC7 = m_SD_precompute_col_ordered.block<9, 4>(0, 56).array().rowwise()*m_F_Transpose_CombinedBlockDamping_col_ordered.block<4, 1>(56, 0).array().transpose();
    ////ChMatrixNMc<double, 9, 4> ChunkDvC8 = m_SD_precompute_col_ordered.block<9, 4>(0, 56).array().rowwise()*m_F_Transpose_CombinedBlockDamping_col_ordered.block<4, 1>(56, 1).array().transpose();
    ////ChMatrixNMc<double, 9, 4> ChunkDvC9 = m_SD_precompute_col_ordered.block<9, 4>(0, 56).array().rowwise()*m_F_Transpose_CombinedBlockDamping_col_ordered.block<4, 1>(56, 2).array().transpose();

    //ChunkDvA1 = m_SD_precompute_col_ordered.block<9, 4>(0, 48).array().rowwise()*m_F_Transpose_CombinedBlockDamping_col_ordered.block<4, 1>(48, 0).array().transpose();
    //ChunkDvA2 = m_SD_precompute_col_ordered.block<9, 4>(0, 48).array().rowwise()*m_F_Transpose_CombinedBlockDamping_col_ordered.block<4, 1>(48, 1).array().transpose();
    //ChunkDvA3 = m_SD_precompute_col_ordered.block<9, 4>(0, 48).array().rowwise()*m_F_Transpose_CombinedBlockDamping_col_ordered.block<4, 1>(48, 2).array().transpose();

    //ChunkDvB4 = m_SD_precompute_col_ordered.block<9, 4>(0, 52).array().rowwise()*m_F_Transpose_CombinedBlockDamping_col_ordered.block<4, 1>(52, 0).array().transpose();
    //ChunkDvB5 = m_SD_precompute_col_ordered.block<9, 4>(0, 52).array().rowwise()*m_F_Transpose_CombinedBlockDamping_col_ordered.block<4, 1>(52, 1).array().transpose();
    //ChunkDvB6 = m_SD_precompute_col_ordered.block<9, 4>(0, 52).array().rowwise()*m_F_Transpose_CombinedBlockDamping_col_ordered.block<4, 1>(52, 2).array().transpose();

    //ChunkDvC7 = m_SD_precompute_col_ordered.block<9, 4>(0, 56).array().rowwise()*m_F_Transpose_CombinedBlockDamping_col_ordered.block<4, 1>(56, 0).array().transpose();
    //ChunkDvC8 = m_SD_precompute_col_ordered.block<9, 4>(0, 56).array().rowwise()*m_F_Transpose_CombinedBlockDamping_col_ordered.block<4, 1>(56, 1).array().transpose();
    //ChunkDvC9 = m_SD_precompute_col_ordered.block<9, 4>(0, 56).array().rowwise()*m_F_Transpose_CombinedBlockDamping_col_ordered.block<4, 1>(56, 2).array().transpose();


    ////ChunkSA1 = SD_reordered(:, (GQPnts)+0 * 16).*repmat((GQW(GQPnts).*(((Kfactor + m_Alpha * Rfactor)*Fcol0_Transpose_D0_Block(GQPnts, 1)) + ((m_Alpha*Kfactor)*Fdotcol0_Transpose_D0_Block(GQPnts, 1))))',NumShapeFunctions,1);
    ////ChunkSA2 = SD_reordered(:, (GQPnts)+0 * 16).*repmat((GQW(GQPnts).*(((Kfactor + m_Alpha * Rfactor)*Fcol0_Transpose_D0_Block(GQPnts, 2)) + ((m_Alpha*Kfactor)*Fdotcol0_Transpose_D0_Block(GQPnts, 2))))',NumShapeFunctions,1);
    ////ChunkSA3 = SD_reordered(:, (GQPnts)+0 * 16).*repmat((GQW(GQPnts).*(((Kfactor + m_Alpha * Rfactor)*Fcol0_Transpose_D0_Block(GQPnts, 3)) + ((m_Alpha*Kfactor)*Fdotcol0_Transpose_D0_Block(GQPnts, 3))))',NumShapeFunctions,1);
    ////ChunkSA4 = SD_reordered(:, (GQPnts)+1 * 16).*repmat((GQW(GQPnts).*(((Kfactor + m_Alpha * Rfactor)*Fcol0_Transpose_D0_Block(GQPnts, 1)) + ((m_Alpha*Kfactor)*Fdotcol0_Transpose_D0_Block(GQPnts, 1))))',NumShapeFunctions,1);
    ////ChunkSA5 = SD_reordered(:, (GQPnts)+1 * 16).*repmat((GQW(GQPnts).*(((Kfactor + m_Alpha * Rfactor)*Fcol0_Transpose_D0_Block(GQPnts, 2)) + ((m_Alpha*Kfactor)*Fdotcol0_Transpose_D0_Block(GQPnts, 2))))',NumShapeFunctions,1);
    ////ChunkSA6 = SD_reordered(:, (GQPnts)+1 * 16).*repmat((GQW(GQPnts).*(((Kfactor + m_Alpha * Rfactor)*Fcol0_Transpose_D0_Block(GQPnts, 3)) + ((m_Alpha*Kfactor)*Fdotcol0_Transpose_D0_Block(GQPnts, 3))))',NumShapeFunctions,1);
    ////ChunkSA7 = SD_reordered(:, (GQPnts)+2 * 16).*repmat((GQW(GQPnts).*(((Kfactor + m_Alpha * Rfactor)*Fcol0_Transpose_D0_Block(GQPnts, 1)) + ((m_Alpha*Kfactor)*Fdotcol0_Transpose_D0_Block(GQPnts, 1))))',NumShapeFunctions,1);
    ////ChunkSA8 = SD_reordered(:, (GQPnts)+2 * 16).*repmat((GQW(GQPnts).*(((Kfactor + m_Alpha * Rfactor)*Fcol0_Transpose_D0_Block(GQPnts, 2)) + ((m_Alpha*Kfactor)*Fdotcol0_Transpose_D0_Block(GQPnts, 2))))',NumShapeFunctions,1);
    ////ChunkSA9 = SD_reordered(:, (GQPnts)+2 * 16).*repmat((GQW(GQPnts).*(((Kfactor + m_Alpha * Rfactor)*Fcol0_Transpose_D0_Block(GQPnts, 3)) + ((m_Alpha*Kfactor)*Fdotcol0_Transpose_D0_Block(GQPnts, 3))))',NumShapeFunctions,1);

    ////ChunkSB1 = SD_reordered(:, (GQPnts)+0 * 16).*repmat((GQW(GQPnts).*(((Kfactor + m_Alpha * Rfactor)*Fcol1_Transpose_D0_Block(GQPnts, 1)) + ((m_Alpha*Kfactor)*Fdotcol1_Transpose_D0_Block(GQPnts, 1))))',NumShapeFunctions,1);
    ////ChunkSB2 = SD_reordered(:, (GQPnts)+0 * 16).*repmat((GQW(GQPnts).*(((Kfactor + m_Alpha * Rfactor)*Fcol1_Transpose_D0_Block(GQPnts, 2)) + ((m_Alpha*Kfactor)*Fdotcol1_Transpose_D0_Block(GQPnts, 2))))',NumShapeFunctions,1);
    ////ChunkSB3 = SD_reordered(:, (GQPnts)+0 * 16).*repmat((GQW(GQPnts).*(((Kfactor + m_Alpha * Rfactor)*Fcol1_Transpose_D0_Block(GQPnts, 3)) + ((m_Alpha*Kfactor)*Fdotcol1_Transpose_D0_Block(GQPnts, 3))))',NumShapeFunctions,1);
    ////ChunkSB4 = SD_reordered(:, (GQPnts)+1 * 16).*repmat((GQW(GQPnts).*(((Kfactor + m_Alpha * Rfactor)*Fcol1_Transpose_D0_Block(GQPnts, 1)) + ((m_Alpha*Kfactor)*Fdotcol1_Transpose_D0_Block(GQPnts, 1))))',NumShapeFunctions,1);
    ////ChunkSB5 = SD_reordered(:, (GQPnts)+1 * 16).*repmat((GQW(GQPnts).*(((Kfactor + m_Alpha * Rfactor)*Fcol1_Transpose_D0_Block(GQPnts, 2)) + ((m_Alpha*Kfactor)*Fdotcol1_Transpose_D0_Block(GQPnts, 2))))',NumShapeFunctions,1);
    ////ChunkSB6 = SD_reordered(:, (GQPnts)+1 * 16).*repmat((GQW(GQPnts).*(((Kfactor + m_Alpha * Rfactor)*Fcol1_Transpose_D0_Block(GQPnts, 3)) + ((m_Alpha*Kfactor)*Fdotcol1_Transpose_D0_Block(GQPnts, 3))))',NumShapeFunctions,1);
    ////ChunkSB7 = SD_reordered(:, (GQPnts)+2 * 16).*repmat((GQW(GQPnts).*(((Kfactor + m_Alpha * Rfactor)*Fcol1_Transpose_D0_Block(GQPnts, 1)) + ((m_Alpha*Kfactor)*Fdotcol1_Transpose_D0_Block(GQPnts, 1))))',NumShapeFunctions,1);
    ////ChunkSB8 = SD_reordered(:, (GQPnts)+2 * 16).*repmat((GQW(GQPnts).*(((Kfactor + m_Alpha * Rfactor)*Fcol1_Transpose_D0_Block(GQPnts, 2)) + ((m_Alpha*Kfactor)*Fdotcol1_Transpose_D0_Block(GQPnts, 2))))',NumShapeFunctions,1);
    ////ChunkSB9 = SD_reordered(:, (GQPnts)+2 * 16).*repmat((GQW(GQPnts).*(((Kfactor + m_Alpha * Rfactor)*Fcol1_Transpose_D0_Block(GQPnts, 3)) + ((m_Alpha*Kfactor)*Fdotcol1_Transpose_D0_Block(GQPnts, 3))))',NumShapeFunctions,1);

    ////ChunkSC1 = SD_reordered(:, (GQPnts)+0 * 16).*repmat((GQW(GQPnts).*(((Kfactor + m_Alpha * Rfactor)*Fcol2_Transpose_D0_Block(GQPnts, 1)) + ((m_Alpha*Kfactor)*Fdotcol2_Transpose_D0_Block(GQPnts, 1))))',NumShapeFunctions,1);
    ////ChunkSC2 = SD_reordered(:, (GQPnts)+0 * 16).*repmat((GQW(GQPnts).*(((Kfactor + m_Alpha * Rfactor)*Fcol2_Transpose_D0_Block(GQPnts, 2)) + ((m_Alpha*Kfactor)*Fdotcol2_Transpose_D0_Block(GQPnts, 2))))',NumShapeFunctions,1);
    ////ChunkSC3 = SD_reordered(:, (GQPnts)+0 * 16).*repmat((GQW(GQPnts).*(((Kfactor + m_Alpha * Rfactor)*Fcol2_Transpose_D0_Block(GQPnts, 3)) + ((m_Alpha*Kfactor)*Fdotcol2_Transpose_D0_Block(GQPnts, 3))))',NumShapeFunctions,1);
    ////ChunkSC4 = SD_reordered(:, (GQPnts)+1 * 16).*repmat((GQW(GQPnts).*(((Kfactor + m_Alpha * Rfactor)*Fcol2_Transpose_D0_Block(GQPnts, 1)) + ((m_Alpha*Kfactor)*Fdotcol2_Transpose_D0_Block(GQPnts, 1))))',NumShapeFunctions,1);
    ////ChunkSC5 = SD_reordered(:, (GQPnts)+1 * 16).*repmat((GQW(GQPnts).*(((Kfactor + m_Alpha * Rfactor)*Fcol2_Transpose_D0_Block(GQPnts, 2)) + ((m_Alpha*Kfactor)*Fdotcol2_Transpose_D0_Block(GQPnts, 2))))',NumShapeFunctions,1);
    ////ChunkSC6 = SD_reordered(:, (GQPnts)+1 * 16).*repmat((GQW(GQPnts).*(((Kfactor + m_Alpha * Rfactor)*Fcol2_Transpose_D0_Block(GQPnts, 3)) + ((m_Alpha*Kfactor)*Fdotcol2_Transpose_D0_Block(GQPnts, 3))))',NumShapeFunctions,1);
    ////ChunkSC7 = SD_reordered(:, (GQPnts)+2 * 16).*repmat((GQW(GQPnts).*(((Kfactor + m_Alpha * Rfactor)*Fcol2_Transpose_D0_Block(GQPnts, 1)) + ((m_Alpha*Kfactor)*Fdotcol2_Transpose_D0_Block(GQPnts, 1))))',NumShapeFunctions,1);
    ////ChunkSC8 = SD_reordered(:, (GQPnts)+2 * 16).*repmat((GQW(GQPnts).*(((Kfactor + m_Alpha * Rfactor)*Fcol2_Transpose_D0_Block(GQPnts, 2)) + ((m_Alpha*Kfactor)*Fdotcol2_Transpose_D0_Block(GQPnts, 2))))',NumShapeFunctions,1);
    ////ChunkSC9 = SD_reordered(:, (GQPnts)+2 * 16).*repmat((GQW(GQPnts).*(((Kfactor + m_Alpha * Rfactor)*Fcol2_Transpose_D0_Block(GQPnts, 3)) + ((m_Alpha*Kfactor)*Fdotcol2_Transpose_D0_Block(GQPnts, 3))))',NumShapeFunctions,1);

    //ChMatrixNMc<double, 9, 16> ChunkSA1 = m_SD_precompute_col_ordered.block<9, 16>(0, 0).array().rowwise()*(((Kfactor + m_Alpha * Rfactor)*m_F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(0, 0).array().transpose()) + ((m_Alpha*Kfactor)*m_F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(0, 3).array().transpose()));
    //ChMatrixNMc<double, 9, 16> ChunkSA2 = m_SD_precompute_col_ordered.block<9, 16>(0, 0).array().rowwise()*(((Kfactor + m_Alpha * Rfactor)*m_F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(0, 1).array().transpose()) + ((m_Alpha*Kfactor)*m_F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(0, 4).array().transpose()));
    //ChMatrixNMc<double, 9, 16> ChunkSA3 = m_SD_precompute_col_ordered.block<9, 16>(0, 0).array().rowwise()*(((Kfactor + m_Alpha * Rfactor)*m_F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(0, 2).array().transpose()) + ((m_Alpha*Kfactor)*m_F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(0, 5).array().transpose()));
    //ChMatrixNMc<double, 9, 16> ChunkSA4 = m_SD_precompute_col_ordered.block<9, 16>(0, 16).array().rowwise()*(((Kfactor + m_Alpha * Rfactor)*m_F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(0, 0).array().transpose()) + ((m_Alpha*Kfactor)*m_F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(0, 3).array().transpose()));
    //ChMatrixNMc<double, 9, 16> ChunkSA5 = m_SD_precompute_col_ordered.block<9, 16>(0, 16).array().rowwise()*(((Kfactor + m_Alpha * Rfactor)*m_F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(0, 1).array().transpose()) + ((m_Alpha*Kfactor)*m_F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(0, 4).array().transpose()));
    //ChMatrixNMc<double, 9, 16> ChunkSA6 = m_SD_precompute_col_ordered.block<9, 16>(0, 16).array().rowwise()*(((Kfactor + m_Alpha * Rfactor)*m_F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(0, 2).array().transpose()) + ((m_Alpha*Kfactor)*m_F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(0, 5).array().transpose()));
    //ChMatrixNMc<double, 9, 16> ChunkSA7 = m_SD_precompute_col_ordered.block<9, 16>(0, 32).array().rowwise()*(((Kfactor + m_Alpha * Rfactor)*m_F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(0, 0).array().transpose()) + ((m_Alpha*Kfactor)*m_F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(0, 3).array().transpose()));
    //ChMatrixNMc<double, 9, 16> ChunkSA8 = m_SD_precompute_col_ordered.block<9, 16>(0, 32).array().rowwise()*(((Kfactor + m_Alpha * Rfactor)*m_F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(0, 1).array().transpose()) + ((m_Alpha*Kfactor)*m_F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(0, 4).array().transpose()));
    //ChMatrixNMc<double, 9, 16> ChunkSA9 = m_SD_precompute_col_ordered.block<9, 16>(0, 32).array().rowwise()*(((Kfactor + m_Alpha * Rfactor)*m_F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(0, 2).array().transpose()) + ((m_Alpha*Kfactor)*m_F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(0, 5).array().transpose()));

    //ChMatrixNMc<double, 9, 16> ChunkSB1 = m_SD_precompute_col_ordered.block<9, 16>(0, 0).array().rowwise()*(((Kfactor + m_Alpha * Rfactor)*m_F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(16, 0).array().transpose()) + ((m_Alpha*Kfactor)*m_F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(16, 3).array().transpose()));
    //ChMatrixNMc<double, 9, 16> ChunkSB2 = m_SD_precompute_col_ordered.block<9, 16>(0, 0).array().rowwise()*(((Kfactor + m_Alpha * Rfactor)*m_F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(16, 1).array().transpose()) + ((m_Alpha*Kfactor)*m_F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(16, 4).array().transpose()));
    //ChMatrixNMc<double, 9, 16> ChunkSB3 = m_SD_precompute_col_ordered.block<9, 16>(0, 0).array().rowwise()*(((Kfactor + m_Alpha * Rfactor)*m_F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(16, 2).array().transpose()) + ((m_Alpha*Kfactor)*m_F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(16, 5).array().transpose()));
    //ChMatrixNMc<double, 9, 16> ChunkSB4 = m_SD_precompute_col_ordered.block<9, 16>(0, 16).array().rowwise()*(((Kfactor + m_Alpha * Rfactor)*m_F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(16, 0).array().transpose()) + ((m_Alpha*Kfactor)*m_F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(16, 3).array().transpose()));
    //ChMatrixNMc<double, 9, 16> ChunkSB5 = m_SD_precompute_col_ordered.block<9, 16>(0, 16).array().rowwise()*(((Kfactor + m_Alpha * Rfactor)*m_F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(16, 1).array().transpose()) + ((m_Alpha*Kfactor)*m_F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(16, 4).array().transpose()));
    //ChMatrixNMc<double, 9, 16> ChunkSB6 = m_SD_precompute_col_ordered.block<9, 16>(0, 16).array().rowwise()*(((Kfactor + m_Alpha * Rfactor)*m_F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(16, 2).array().transpose()) + ((m_Alpha*Kfactor)*m_F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(16, 5).array().transpose()));
    //ChMatrixNMc<double, 9, 16> ChunkSB7 = m_SD_precompute_col_ordered.block<9, 16>(0, 32).array().rowwise()*(((Kfactor + m_Alpha * Rfactor)*m_F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(16, 0).array().transpose()) + ((m_Alpha*Kfactor)*m_F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(16, 3).array().transpose()));
    //ChMatrixNMc<double, 9, 16> ChunkSB8 = m_SD_precompute_col_ordered.block<9, 16>(0, 32).array().rowwise()*(((Kfactor + m_Alpha * Rfactor)*m_F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(16, 1).array().transpose()) + ((m_Alpha*Kfactor)*m_F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(16, 4).array().transpose()));
    //ChMatrixNMc<double, 9, 16> ChunkSB9 = m_SD_precompute_col_ordered.block<9, 16>(0, 32).array().rowwise()*(((Kfactor + m_Alpha * Rfactor)*m_F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(16, 2).array().transpose()) + ((m_Alpha*Kfactor)*m_F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(16, 5).array().transpose()));

    //ChMatrixNMc<double, 9, 16> ChunkSC1 = m_SD_precompute_col_ordered.block<9, 16>(0, 0).array().rowwise()*(((Kfactor + m_Alpha * Rfactor)*m_F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(32, 0).array().transpose()) + ((m_Alpha*Kfactor)*m_F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(32, 3).array().transpose()));
    //ChMatrixNMc<double, 9, 16> ChunkSC2 = m_SD_precompute_col_ordered.block<9, 16>(0, 0).array().rowwise()*(((Kfactor + m_Alpha * Rfactor)*m_F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(32, 1).array().transpose()) + ((m_Alpha*Kfactor)*m_F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(32, 4).array().transpose()));
    //ChMatrixNMc<double, 9, 16> ChunkSC3 = m_SD_precompute_col_ordered.block<9, 16>(0, 0).array().rowwise()*(((Kfactor + m_Alpha * Rfactor)*m_F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(32, 2).array().transpose()) + ((m_Alpha*Kfactor)*m_F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(32, 5).array().transpose()));
    //ChMatrixNMc<double, 9, 16> ChunkSC4 = m_SD_precompute_col_ordered.block<9, 16>(0, 16).array().rowwise()*(((Kfactor + m_Alpha * Rfactor)*m_F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(32, 0).array().transpose()) + ((m_Alpha*Kfactor)*m_F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(32, 3).array().transpose()));
    //ChMatrixNMc<double, 9, 16> ChunkSC5 = m_SD_precompute_col_ordered.block<9, 16>(0, 16).array().rowwise()*(((Kfactor + m_Alpha * Rfactor)*m_F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(32, 1).array().transpose()) + ((m_Alpha*Kfactor)*m_F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(32, 4).array().transpose()));
    //ChMatrixNMc<double, 9, 16> ChunkSC6 = m_SD_precompute_col_ordered.block<9, 16>(0, 16).array().rowwise()*(((Kfactor + m_Alpha * Rfactor)*m_F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(32, 2).array().transpose()) + ((m_Alpha*Kfactor)*m_F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(32, 5).array().transpose()));
    //ChMatrixNMc<double, 9, 16> ChunkSC7 = m_SD_precompute_col_ordered.block<9, 16>(0, 32).array().rowwise()*(((Kfactor + m_Alpha * Rfactor)*m_F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(32, 0).array().transpose()) + ((m_Alpha*Kfactor)*m_F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(32, 3).array().transpose()));
    //ChMatrixNMc<double, 9, 16> ChunkSC8 = m_SD_precompute_col_ordered.block<9, 16>(0, 32).array().rowwise()*(((Kfactor + m_Alpha * Rfactor)*m_F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(32, 1).array().transpose()) + ((m_Alpha*Kfactor)*m_F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(32, 4).array().transpose()));
    //ChMatrixNMc<double, 9, 16> ChunkSC9 = m_SD_precompute_col_ordered.block<9, 16>(0, 32).array().rowwise()*(((Kfactor + m_Alpha * Rfactor)*m_F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(32, 2).array().transpose()) + ((m_Alpha*Kfactor)*m_F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(32, 5).array().transpose()));


    ////ChunkDvSA1 = SD_reordered(:, (GQPnts)+2 * 16 + 0).*repmat((GQW(GQPnts).*(((Kfactor + m_Alpha * Rfactor)*Fcol0_Transpose_Dv_Block(GQPnts - 16, 1)) + ((m_Alpha*Kfactor)*Fdotcol0_Transpose_Dv_Block(GQPnts - 16, 1))))',NumShapeFunctions,1);
    ////ChunkDvSA2 = SD_reordered(:, (GQPnts)+2 * 16 + 0).*repmat((GQW(GQPnts).*(((Kfactor + m_Alpha * Rfactor)*Fcol0_Transpose_Dv_Block(GQPnts - 16, 2)) + ((m_Alpha*Kfactor)*Fdotcol0_Transpose_Dv_Block(GQPnts - 16, 2))))',NumShapeFunctions,1);
    ////ChunkDvSA3 = SD_reordered(:, (GQPnts)+2 * 16 + 0).*repmat((GQW(GQPnts).*(((Kfactor + m_Alpha * Rfactor)*Fcol0_Transpose_Dv_Block(GQPnts - 16, 3)) + ((m_Alpha*Kfactor)*Fdotcol0_Transpose_Dv_Block(GQPnts - 16, 3))))',NumShapeFunctions,1);
    ////
    ////ChunkDvSB4 = SD_reordered(:, (GQPnts)+2 * 16 + 4).*repmat((GQW(GQPnts).*(((Kfactor + m_Alpha * Rfactor)*Fcol1_Transpose_Dv_Block(GQPnts - 16, 1)) + ((m_Alpha*Kfactor)*Fdotcol1_Transpose_Dv_Block(GQPnts - 16, 1))))',NumShapeFunctions,1);
    ////ChunkDvSB5 = SD_reordered(:, (GQPnts)+2 * 16 + 4).*repmat((GQW(GQPnts).*(((Kfactor + m_Alpha * Rfactor)*Fcol1_Transpose_Dv_Block(GQPnts - 16, 2)) + ((m_Alpha*Kfactor)*Fdotcol1_Transpose_Dv_Block(GQPnts - 16, 2))))',NumShapeFunctions,1);
    ////ChunkDvSB6 = SD_reordered(:, (GQPnts)+2 * 16 + 4).*repmat((GQW(GQPnts).*(((Kfactor + m_Alpha * Rfactor)*Fcol1_Transpose_Dv_Block(GQPnts - 16, 3)) + ((m_Alpha*Kfactor)*Fdotcol1_Transpose_Dv_Block(GQPnts - 16, 3))))',NumShapeFunctions,1);
    ////
    ////ChunkDvSC7 = SD_reordered(:, (GQPnts)+2 * 16 + 8).*repmat((GQW(GQPnts).*(((Kfactor + m_Alpha * Rfactor)*Fcol2_Transpose_Dv_Block(GQPnts - 16, 1)) + ((m_Alpha*Kfactor)*Fdotcol2_Transpose_Dv_Block(GQPnts - 16, 1))))',NumShapeFunctions,1);
    ////ChunkDvSC8 = SD_reordered(:, (GQPnts)+2 * 16 + 8).*repmat((GQW(GQPnts).*(((Kfactor + m_Alpha * Rfactor)*Fcol2_Transpose_Dv_Block(GQPnts - 16, 2)) + ((m_Alpha*Kfactor)*Fdotcol2_Transpose_Dv_Block(GQPnts - 16, 2))))',NumShapeFunctions,1);
    ////ChunkDvSC9 = SD_reordered(:, (GQPnts)+2 * 16 + 8).*repmat((GQW(GQPnts).*(((Kfactor + m_Alpha * Rfactor)*Fcol2_Transpose_Dv_Block(GQPnts - 16, 3)) + ((m_Alpha*Kfactor)*Fdotcol2_Transpose_Dv_Block(GQPnts - 16, 3))))',NumShapeFunctions,1);

    ////ChMatrixNMc<double, 9, 4> ChunkDvSA1 = m_SD_precompute_col_ordered.block<9, 4>(0, 48).array().rowwise()*(((Kfactor + m_Alpha * Rfactor)*m_F_Transpose_CombinedBlockDamping_col_ordered.block<4, 1>(48, 0).array().transpose()) + ((m_Alpha*Kfactor)*m_F_Transpose_CombinedBlockDamping_col_ordered.block<4, 1>(48, 3).array().transpose()));
    ////ChMatrixNMc<double, 9, 4> ChunkDvSA2 = m_SD_precompute_col_ordered.block<9, 4>(0, 48).array().rowwise()*(((Kfactor + m_Alpha * Rfactor)*m_F_Transpose_CombinedBlockDamping_col_ordered.block<4, 1>(48, 1).array().transpose()) + ((m_Alpha*Kfactor)*m_F_Transpose_CombinedBlockDamping_col_ordered.block<4, 1>(48, 4).array().transpose()));
    ////ChMatrixNMc<double, 9, 4> ChunkDvSA3 = m_SD_precompute_col_ordered.block<9, 4>(0, 48).array().rowwise()*(((Kfactor + m_Alpha * Rfactor)*m_F_Transpose_CombinedBlockDamping_col_ordered.block<4, 1>(48, 2).array().transpose()) + ((m_Alpha*Kfactor)*m_F_Transpose_CombinedBlockDamping_col_ordered.block<4, 1>(48, 5).array().transpose()));

    ////ChMatrixNMc<double, 9, 4> ChunkDvSB4 = m_SD_precompute_col_ordered.block<9, 4>(0, 52).array().rowwise()*(((Kfactor + m_Alpha * Rfactor)*m_F_Transpose_CombinedBlockDamping_col_ordered.block<4, 1>(52, 0).array().transpose()) + ((m_Alpha*Kfactor)*m_F_Transpose_CombinedBlockDamping_col_ordered.block<4, 1>(52, 3).array().transpose()));
    ////ChMatrixNMc<double, 9, 4> ChunkDvSB5 = m_SD_precompute_col_ordered.block<9, 4>(0, 52).array().rowwise()*(((Kfactor + m_Alpha * Rfactor)*m_F_Transpose_CombinedBlockDamping_col_ordered.block<4, 1>(52, 1).array().transpose()) + ((m_Alpha*Kfactor)*m_F_Transpose_CombinedBlockDamping_col_ordered.block<4, 1>(52, 4).array().transpose()));
    ////ChMatrixNMc<double, 9, 4> ChunkDvSB6 = m_SD_precompute_col_ordered.block<9, 4>(0, 52).array().rowwise()*(((Kfactor + m_Alpha * Rfactor)*m_F_Transpose_CombinedBlockDamping_col_ordered.block<4, 1>(52, 2).array().transpose()) + ((m_Alpha*Kfactor)*m_F_Transpose_CombinedBlockDamping_col_ordered.block<4, 1>(52, 5).array().transpose()));

    ////ChMatrixNMc<double, 9, 4> ChunkDvSC7 = m_SD_precompute_col_ordered.block<9, 4>(0, 56).array().rowwise()*(((Kfactor + m_Alpha * Rfactor)*m_F_Transpose_CombinedBlockDamping_col_ordered.block<4, 1>(56, 0).array().transpose()) + ((m_Alpha*Kfactor)*m_F_Transpose_CombinedBlockDamping_col_ordered.block<4, 1>(56, 3).array().transpose()));
    ////ChMatrixNMc<double, 9, 4> ChunkDvSC8 = m_SD_precompute_col_ordered.block<9, 4>(0, 56).array().rowwise()*(((Kfactor + m_Alpha * Rfactor)*m_F_Transpose_CombinedBlockDamping_col_ordered.block<4, 1>(56, 1).array().transpose()) + ((m_Alpha*Kfactor)*m_F_Transpose_CombinedBlockDamping_col_ordered.block<4, 1>(56, 4).array().transpose()));
    ////ChMatrixNMc<double, 9, 4> ChunkDvSC9 = m_SD_precompute_col_ordered.block<9, 4>(0, 56).array().rowwise()*(((Kfactor + m_Alpha * Rfactor)*m_F_Transpose_CombinedBlockDamping_col_ordered.block<4, 1>(56, 2).array().transpose()) + ((m_Alpha*Kfactor)*m_F_Transpose_CombinedBlockDamping_col_ordered.block<4, 1>(56, 5).array().transpose()));

    //Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor, 9, 4> ChunkDvSA1;
    //ChunkDvSA1.resize(9, 4);
    //Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor, 9, 4> ChunkDvSA2;
    //ChunkDvSA2.resize(9, 4);
    //Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor, 9, 4> ChunkDvSA3;
    //ChunkDvSA3.resize(9, 4);

    //Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor, 9, 4> ChunkDvSB4;
    //ChunkDvSB4.resize(9, 4);
    //Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor, 9, 4> ChunkDvSB5;
    //ChunkDvSB5.resize(9, 4);
    //Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor, 9, 4> ChunkDvSB6;
    //ChunkDvSB6.resize(9, 4);

    //Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor, 9, 4> ChunkDvSC7;
    //ChunkDvSC7.resize(9, 4);
    //Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor, 9, 4> ChunkDvSC8;
    //ChunkDvSC8.resize(9, 4);
    //Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor, 9, 4> ChunkDvSC9;
    //ChunkDvSC9.resize(9, 4);

    //ChunkDvSA1 = m_SD_precompute_col_ordered.block<9, 4>(0, 48).array().rowwise()*(((Kfactor + m_Alpha * Rfactor)*m_F_Transpose_CombinedBlockDamping_col_ordered.block<4, 1>(48, 0).array().transpose()) + ((m_Alpha*Kfactor)*m_F_Transpose_CombinedBlockDamping_col_ordered.block<4, 1>(48, 3).array().transpose()));
    //ChunkDvSA2 = m_SD_precompute_col_ordered.block<9, 4>(0, 48).array().rowwise()*(((Kfactor + m_Alpha * Rfactor)*m_F_Transpose_CombinedBlockDamping_col_ordered.block<4, 1>(48, 1).array().transpose()) + ((m_Alpha*Kfactor)*m_F_Transpose_CombinedBlockDamping_col_ordered.block<4, 1>(48, 4).array().transpose()));
    //ChunkDvSA3 = m_SD_precompute_col_ordered.block<9, 4>(0, 48).array().rowwise()*(((Kfactor + m_Alpha * Rfactor)*m_F_Transpose_CombinedBlockDamping_col_ordered.block<4, 1>(48, 2).array().transpose()) + ((m_Alpha*Kfactor)*m_F_Transpose_CombinedBlockDamping_col_ordered.block<4, 1>(48, 5).array().transpose()));

    //ChunkDvSB4 = m_SD_precompute_col_ordered.block<9, 4>(0, 52).array().rowwise()*(((Kfactor + m_Alpha * Rfactor)*m_F_Transpose_CombinedBlockDamping_col_ordered.block<4, 1>(52, 0).array().transpose()) + ((m_Alpha*Kfactor)*m_F_Transpose_CombinedBlockDamping_col_ordered.block<4, 1>(52, 3).array().transpose()));
    //ChunkDvSB5 = m_SD_precompute_col_ordered.block<9, 4>(0, 52).array().rowwise()*(((Kfactor + m_Alpha * Rfactor)*m_F_Transpose_CombinedBlockDamping_col_ordered.block<4, 1>(52, 1).array().transpose()) + ((m_Alpha*Kfactor)*m_F_Transpose_CombinedBlockDamping_col_ordered.block<4, 1>(52, 4).array().transpose()));
    //ChunkDvSB6 = m_SD_precompute_col_ordered.block<9, 4>(0, 52).array().rowwise()*(((Kfactor + m_Alpha * Rfactor)*m_F_Transpose_CombinedBlockDamping_col_ordered.block<4, 1>(52, 2).array().transpose()) + ((m_Alpha*Kfactor)*m_F_Transpose_CombinedBlockDamping_col_ordered.block<4, 1>(52, 5).array().transpose()));

    //ChunkDvSC7 = m_SD_precompute_col_ordered.block<9, 4>(0, 56).array().rowwise()*(((Kfactor + m_Alpha * Rfactor)*m_F_Transpose_CombinedBlockDamping_col_ordered.block<4, 1>(56, 0).array().transpose()) + ((m_Alpha*Kfactor)*m_F_Transpose_CombinedBlockDamping_col_ordered.block<4, 1>(56, 3).array().transpose()));
    //ChunkDvSC8 = m_SD_precompute_col_ordered.block<9, 4>(0, 56).array().rowwise()*(((Kfactor + m_Alpha * Rfactor)*m_F_Transpose_CombinedBlockDamping_col_ordered.block<4, 1>(56, 1).array().transpose()) + ((m_Alpha*Kfactor)*m_F_Transpose_CombinedBlockDamping_col_ordered.block<4, 1>(56, 4).array().transpose()));
    //ChunkDvSC9 = m_SD_precompute_col_ordered.block<9, 4>(0, 56).array().rowwise()*(((Kfactor + m_Alpha * Rfactor)*m_F_Transpose_CombinedBlockDamping_col_ordered.block<4, 1>(56, 2).array().transpose()) + ((m_Alpha*Kfactor)*m_F_Transpose_CombinedBlockDamping_col_ordered.block<4, 1>(56, 5).array().transpose()));


    ////ChMatrixNMc<double, 108, 9> partial_epsilon_partial_e_Block11;
    //Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor, 108, 9> partial_epsilon_partial_e_Block11;
    //partial_epsilon_partial_e_Block11.resize(108, 9);
    //partial_epsilon_partial_e_Block11.block<16, 9>(0, 0) = ChunkA1.transpose();
    //partial_epsilon_partial_e_Block11.block<16, 9>(16, 0) = ChunkB4.transpose();
    //partial_epsilon_partial_e_Block11.block<16, 9>(32, 0) = ChunkC7.transpose();
    //partial_epsilon_partial_e_Block11.block<16, 9>(48, 0) = ChunkB7.transpose()+ ChunkC4.transpose();
    //partial_epsilon_partial_e_Block11.block<16, 9>(64, 0) = ChunkA7.transpose() + ChunkC1.transpose();
    //partial_epsilon_partial_e_Block11.block<16, 9>(80, 0) = ChunkA4.transpose() + ChunkB1.transpose();
    //partial_epsilon_partial_e_Block11.block<4, 9>(96, 0) = ChunkDvA1.transpose();
    //partial_epsilon_partial_e_Block11.block<4, 9>(100, 0) = ChunkDvB4.transpose();
    //partial_epsilon_partial_e_Block11.block<4, 9>(104, 0) = ChunkDvC7.transpose();

    ////ChMatrixNMc<double, 108, 9> partial_epsilon_partial_e_Block22;
    //Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor, 108, 9> partial_epsilon_partial_e_Block22;
    //partial_epsilon_partial_e_Block22.resize(108, 9);
    //partial_epsilon_partial_e_Block22.block<16, 9>(0, 0) = ChunkA2.transpose();
    //partial_epsilon_partial_e_Block22.block<16, 9>(16, 0) = ChunkB5.transpose();
    //partial_epsilon_partial_e_Block22.block<16, 9>(32, 0) = ChunkC8.transpose();
    //partial_epsilon_partial_e_Block22.block<16, 9>(48, 0) = ChunkB8.transpose() + ChunkC5.transpose();
    //partial_epsilon_partial_e_Block22.block<16, 9>(64, 0) = ChunkA8.transpose() + ChunkC2.transpose();
    //partial_epsilon_partial_e_Block22.block<16, 9>(80, 0) = ChunkA5.transpose() + ChunkB2.transpose();
    //partial_epsilon_partial_e_Block22.block<4, 9>(96, 0) = ChunkDvA2.transpose();
    //partial_epsilon_partial_e_Block22.block<4, 9>(100, 0) = ChunkDvB5.transpose();
    //partial_epsilon_partial_e_Block22.block<4, 9>(104, 0) = ChunkDvC8.transpose();

    ////ChMatrixNMc<double, 108, 9> partial_epsilon_partial_e_Block33;
    //Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor, 108, 9> partial_epsilon_partial_e_Block33;
    //partial_epsilon_partial_e_Block33.resize(108, 9);
    //partial_epsilon_partial_e_Block33.block<16, 9>(0, 0) = ChunkA3.transpose();
    //partial_epsilon_partial_e_Block33.block<16, 9>(16, 0) = ChunkB6.transpose();
    //partial_epsilon_partial_e_Block33.block<16, 9>(32, 0) = ChunkC9.transpose();
    //partial_epsilon_partial_e_Block33.block<16, 9>(48, 0) = ChunkB9.transpose() + ChunkC6.transpose();
    //partial_epsilon_partial_e_Block33.block<16, 9>(64, 0) = ChunkA9.transpose() + ChunkC3.transpose();
    //partial_epsilon_partial_e_Block33.block<16, 9>(80, 0) = ChunkA6.transpose() + ChunkB3.transpose();
    //partial_epsilon_partial_e_Block33.block<4, 9>(96, 0) = ChunkDvA3.transpose();
    //partial_epsilon_partial_e_Block33.block<4, 9>(100, 0) = ChunkDvB6.transpose();
    //partial_epsilon_partial_e_Block33.block<4, 9>(104, 0) = ChunkDvC9.transpose();

    ////ChMatrixNMc<double, 108, 9> Scaled_Combined_partial_epsilon_partial_e_Block11;
    //Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor, 108, 9> Scaled_Combined_partial_epsilon_partial_e_Block11;
    //Scaled_Combined_partial_epsilon_partial_e_Block11.resize(108, 9);
    //Scaled_Combined_partial_epsilon_partial_e_Block11.block<16, 9>(0, 0) = D0(0)*ChunkSA1.transpose();
    //Scaled_Combined_partial_epsilon_partial_e_Block11.block<16, 9>(16, 0) = D0(1)*ChunkSB4.transpose();
    //Scaled_Combined_partial_epsilon_partial_e_Block11.block<16, 9>(32, 0) = D0(2)*ChunkSC7.transpose();
    //Scaled_Combined_partial_epsilon_partial_e_Block11.block<16, 9>(48, 0) = D0(3)*(ChunkSB7.transpose() + ChunkSC4.transpose());
    //Scaled_Combined_partial_epsilon_partial_e_Block11.block<16, 9>(64, 0) = D0(4)*(ChunkSA7.transpose() + ChunkSC1.transpose());
    //Scaled_Combined_partial_epsilon_partial_e_Block11.block<16, 9>(80, 0) = D0(5)*(ChunkSA4.transpose() + ChunkSB1.transpose());
    //Scaled_Combined_partial_epsilon_partial_e_Block11.block<4, 9>(96, 0) = Dv(0, 0)*ChunkDvSA1.transpose() + Dv(0, 1)*ChunkDvSB4.transpose() + Dv(0, 2)*ChunkDvSC7.transpose();
    //Scaled_Combined_partial_epsilon_partial_e_Block11.block<4, 9>(100, 0) = Dv(1, 0)*ChunkDvSA1.transpose() + Dv(1, 1)*ChunkDvSB4.transpose() + Dv(1, 2)*ChunkDvSC7.transpose();
    //Scaled_Combined_partial_epsilon_partial_e_Block11.block<4, 9>(104, 0) = Dv(2, 0)*ChunkDvSA1.transpose() + Dv(2, 1)*ChunkDvSB4.transpose() + Dv(2, 2)*ChunkDvSC7.transpose();

    ////ChMatrixNMc<double, 108, 9> Scaled_Combined_partial_epsilon_partial_e_Block22;
    //Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor, 108, 9> Scaled_Combined_partial_epsilon_partial_e_Block22;
    //Scaled_Combined_partial_epsilon_partial_e_Block22.resize(108, 9);
    //Scaled_Combined_partial_epsilon_partial_e_Block22.block<16, 9>(0, 0) = D0(0)*ChunkSA2.transpose();
    //Scaled_Combined_partial_epsilon_partial_e_Block22.block<16, 9>(16, 0) = D0(1)*ChunkSB5.transpose();
    //Scaled_Combined_partial_epsilon_partial_e_Block22.block<16, 9>(32, 0) = D0(2)*ChunkSC8.transpose();
    //Scaled_Combined_partial_epsilon_partial_e_Block22.block<16, 9>(48, 0) = D0(3)*(ChunkSB8.transpose() + ChunkSC5.transpose());
    //Scaled_Combined_partial_epsilon_partial_e_Block22.block<16, 9>(64, 0) = D0(4)*(ChunkSA8.transpose() + ChunkSC2.transpose());
    //Scaled_Combined_partial_epsilon_partial_e_Block22.block<16, 9>(80, 0) = D0(5)*(ChunkSA5.transpose() + ChunkSB2.transpose());
    //Scaled_Combined_partial_epsilon_partial_e_Block22.block<4, 9>(96, 0) = Dv(0, 0)*ChunkDvSA2.transpose() + Dv(0, 1)*ChunkDvSB5.transpose() + Dv(0, 2)*ChunkDvSC8.transpose();
    //Scaled_Combined_partial_epsilon_partial_e_Block22.block<4, 9>(100, 0) = Dv(1, 0)*ChunkDvSA2.transpose() + Dv(1, 1)*ChunkDvSB5.transpose() + Dv(1, 2)*ChunkDvSC8.transpose();
    //Scaled_Combined_partial_epsilon_partial_e_Block22.block<4, 9>(104, 0) = Dv(2, 0)*ChunkDvSA2.transpose() + Dv(2, 1)*ChunkDvSB5.transpose() + Dv(2, 2)*ChunkDvSC8.transpose();

    ////ChMatrixNMc<double, 108, 9> Scaled_Combined_partial_epsilon_partial_e_Block33;
    //Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor, 108, 9> Scaled_Combined_partial_epsilon_partial_e_Block33;
    //Scaled_Combined_partial_epsilon_partial_e_Block33.resize(108, 9);
    //Scaled_Combined_partial_epsilon_partial_e_Block33.block<16, 9>(0, 0) = D0(0)*ChunkSA3.transpose();
    //Scaled_Combined_partial_epsilon_partial_e_Block33.block<16, 9>(16, 0) = D0(1)*ChunkSB6.transpose();
    //Scaled_Combined_partial_epsilon_partial_e_Block33.block<16, 9>(32, 0) = D0(2)*ChunkSC9.transpose();
    //Scaled_Combined_partial_epsilon_partial_e_Block33.block<16, 9>(48, 0) = D0(3)*ChunkSB9.transpose() + D0(3)*ChunkSC6.transpose();
    //Scaled_Combined_partial_epsilon_partial_e_Block33.block<16, 9>(64, 0) = D0(4)*ChunkSA9.transpose() + D0(4)*ChunkSC3.transpose();
    //Scaled_Combined_partial_epsilon_partial_e_Block33.block<16, 9>(80, 0) = D0(5)*ChunkSA6.transpose() + D0(5)*ChunkSB3.transpose();
    //Scaled_Combined_partial_epsilon_partial_e_Block33.block<4, 9>(96, 0) = Dv(0, 0)*ChunkDvSA3.transpose() + Dv(0, 1)*ChunkDvSB6.transpose() + Dv(0, 2)*ChunkDvSC9.transpose();
    //Scaled_Combined_partial_epsilon_partial_e_Block33.block<4, 9>(100, 0) = Dv(1, 0)*ChunkDvSA3.transpose() + Dv(1, 1)*ChunkDvSB6.transpose() + Dv(1, 2)*ChunkDvSC9.transpose();
    //Scaled_Combined_partial_epsilon_partial_e_Block33.block<4, 9>(104, 0) = Dv(2, 0)*ChunkDvSA3.transpose() + Dv(2, 1)*ChunkDvSB6.transpose() + Dv(2, 2)*ChunkDvSC9.transpose();


    //Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor, 9, 9> Block11;
    //Block11.resize(9, 9);
    //Block11 = partial_epsilon_partial_e_Block11.transpose()*Scaled_Combined_partial_epsilon_partial_e_Block11;
    //Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor, 9, 9> Block12;
    //Block12.resize(9, 9);
    //Block12 = partial_epsilon_partial_e_Block11.transpose()*Scaled_Combined_partial_epsilon_partial_e_Block22;
    //Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor, 9, 9> Block13;
    //Block13.resize(9, 9);
    //Block13 = partial_epsilon_partial_e_Block11.transpose()*Scaled_Combined_partial_epsilon_partial_e_Block33;

    //Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor, 9, 9> Block21;
    //Block21.resize(9, 9);
    //Block21 = partial_epsilon_partial_e_Block22.transpose()*Scaled_Combined_partial_epsilon_partial_e_Block11;
    //Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor, 9, 9> Block22;
    //Block22.resize(9, 9);
    //Block22 = partial_epsilon_partial_e_Block22.transpose()*Scaled_Combined_partial_epsilon_partial_e_Block22;
    //Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor, 9, 9> Block23;
    //Block23.resize(9, 9);
    //Block23 = partial_epsilon_partial_e_Block22.transpose()*Scaled_Combined_partial_epsilon_partial_e_Block33;

    //Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor, 9, 9> Block31;
    //Block31.resize(9, 9);
    //Block31 = partial_epsilon_partial_e_Block33.transpose()*Scaled_Combined_partial_epsilon_partial_e_Block11;
    //Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor, 9, 9> Block32;
    //Block32.resize(9, 9);
    //Block32 = partial_epsilon_partial_e_Block33.transpose()*Scaled_Combined_partial_epsilon_partial_e_Block22;
    //Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor, 9, 9> Block33;
    //Block33.resize(9, 9);
    //Block33 = partial_epsilon_partial_e_Block33.transpose()*Scaled_Combined_partial_epsilon_partial_e_Block33;


    ////ChMatrixNMc<double, 9, 9> Block11 = partial_epsilon_partial_e_Block11.transpose()*Scaled_Combined_partial_epsilon_partial_e_Block11;
    ////ChMatrixNMc<double, 9, 9> Block12 = partial_epsilon_partial_e_Block11.transpose()*Scaled_Combined_partial_epsilon_partial_e_Block22;
    ////ChMatrixNMc<double, 9, 9> Block13 = partial_epsilon_partial_e_Block11.transpose()*Scaled_Combined_partial_epsilon_partial_e_Block33;

    ////ChMatrixNMc<double, 9, 9> Block21 = partial_epsilon_partial_e_Block22.transpose()*Scaled_Combined_partial_epsilon_partial_e_Block11;
    ////ChMatrixNMc<double, 9, 9> Block22 = partial_epsilon_partial_e_Block22.transpose()*Scaled_Combined_partial_epsilon_partial_e_Block22;
    ////ChMatrixNMc<double, 9, 9> Block23 = partial_epsilon_partial_e_Block22.transpose()*Scaled_Combined_partial_epsilon_partial_e_Block33;

    ////ChMatrixNMc<double, 9, 9> Block31 = partial_epsilon_partial_e_Block33.transpose()*Scaled_Combined_partial_epsilon_partial_e_Block11;
    ////ChMatrixNMc<double, 9, 9> Block32 = partial_epsilon_partial_e_Block33.transpose()*Scaled_Combined_partial_epsilon_partial_e_Block22;
    ////ChMatrixNMc<double, 9, 9> Block33 = partial_epsilon_partial_e_Block33.transpose()*Scaled_Combined_partial_epsilon_partial_e_Block33;


    ////Inflate the Mass Matrix since it is stored in compact form.
    ////In MATLAB notation:	 
    ////H(1:3:end,1:3:end) = H(1:3:end,1:3:end) + Jacobian_CompactPart;
    ////H(2:3:end,2:3:end) = H(2:3:end,2:3:end) + Jacobian_CompactPart;
    ////H(3:3:end,3:3:end) = H(3:3:end,3:3:end) + Jacobian_CompactPart;
    //for (unsigned int i = 0; i < 9; i++) {
    //    for (unsigned int j = 0; j < 9; j++) {
    //        H(3 * i, 3 * j) = -Block11(i,j) + Jacobian_CompactPart(i, j);
    //        H(3 * i + 1, 3 * j) = -Block21(i, j);
    //        H(3 * i + 2, 3 * j) = -Block31(i, j);

    //        H(3 * i, 3 * j + 1) = -Block12(i, j);
    //        H(3 * i + 1, 3 * j + 1) = -Block22(i, j) + Jacobian_CompactPart(i, j);
    //        H(3 * i + 2, 3 * j + 1) = -Block32(i, j);

    //        H(3 * i, 3 * j + 2) = -Block13(i, j);
    //        H(3 * i + 1, 3 * j + 2) = -Block23(i, j);
    //        H(3 * i + 2, 3 * j + 2) = -Block33(i, j) + Jacobian_CompactPart(i, j);
    //    }
    //}

    ////// Calculate the linear combination Kfactor*[K] + Rfactor*[R]
    ////ChMatrixNM<double, 27, 27> JacobianMatrix;
    ////ComputeInternalJacobians(JacobianMatrix, Kfactor, Rfactor);
    ////ChMatrixNM<double, 27, 27> JacobianMatrix_Analytic = H;
    //////ComputeInternalJacobiansAnalytic(JacobianMatrix_Analytic, Kfactor, Rfactor);
    ////ChMatrixNM<double, 27, 27> Delta = JacobianMatrix_Analytic - JacobianMatrix;
    ////Delta = Delta.cwiseAbs();

    ////std::cout << std::endl << std::endl << "---------------------------" << std::endl << std::endl;
    ////std::cout << "Numeric Jacobian = " << std::endl;
    ////std::cout << JacobianMatrix << std::endl;

    ////std::cout << std::endl << std::endl << "---------------------------" << std::endl << std::endl;
    ////std::cout << "Analytic Jacobian = " << std::endl;
    ////std::cout << JacobianMatrix_Analytic << std::endl;

    ////std::cout << std::endl << std::endl << "---------------------------" << std::endl << std::endl;
    ////std::cout << "Analytic Jacobian - Numeric Jacobian = " << std::endl;
    ////std::cout << JacobianMatrix_Analytic - JacobianMatrix << std::endl;

    ////std::cout << std::endl << std::endl << "---------------------------" << std::endl << std::endl;
    ////std::cout << "Max Delta = " << Delta.colwise().maxCoeff() << std::endl;

    //////ComputeInternalJacobiansAnalytic(JacobianMatrix, Kfactor, Rfactor);

    ////// Load Jac + Mfactor*[M] into H
    ////H = JacobianMatrix;
    ////for (unsigned int i = 0; i < 9; i++) {
    ////    for (unsigned int j = 0; j < 9; j++) {
    ////        H(3 * i, 3 * j) += Mfactor * m_MassMatrix(i, j);
    ////        H(3 * i + 1, 3 * j + 1) += Mfactor * m_MassMatrix(i, j);
    ////        H(3 * i + 2, 3 * j + 2) += Mfactor * m_MassMatrix(i, j);
    ////    }
    ////}

#else
    ChMatrixNM<double, 27, 27> JacobianMatrix;
    ComputeInternalJacobians(JacobianMatrix, Kfactor, Rfactor);
    // Load Jac + Mfactor*[M] into H
    H = JacobianMatrix;
    for (unsigned int i = 0; i < 9; i++) {
        for (unsigned int j = 0; j < 9; j++) {
            H(3 * i, 3 * j) += Mfactor * m_MassMatrix(i, j);
            H(3 * i + 1, 3 * j + 1) += Mfactor * m_MassMatrix(i, j);
            H(3 * i + 2, 3 * j + 2) += Mfactor * m_MassMatrix(i, j);
        }
    }
#endif // true

}

//Calculate the calculate the Jacobian of the internal force integrand with damping included
void ChElementBeamANCF_MT35::ComputeInternalJacobianDamping(ChMatrixRef H, double Kfactor, double Rfactor, double Mfactor) {

    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor, 9, 60> S_scaled_SD_precompute_col_ordered;
    S_scaled_SD_precompute_col_ordered.resize(9, 60);

    for (auto i = 0; i < 9; i++) {
        S_scaled_SD_precompute_col_ordered.block<1, 16>(i, 0) = m_SPK2_0_D0_Block.transpose().cwiseProduct(m_SD_precompute_col_ordered.block<1, 16>(i, 0));
        S_scaled_SD_precompute_col_ordered.block<1, 16>(i, 0) += m_SPK2_5_D0_Block.transpose().cwiseProduct(m_SD_precompute_col_ordered.block<1, 16>(i, 16));
        S_scaled_SD_precompute_col_ordered.block<1, 16>(i, 0) += m_SPK2_4_D0_Block.transpose().cwiseProduct(m_SD_precompute_col_ordered.block<1, 16>(i, 32));

        S_scaled_SD_precompute_col_ordered.block<1, 16>(i, 16) = m_SPK2_5_D0_Block.transpose().cwiseProduct(m_SD_precompute_col_ordered.block<1, 16>(i, 0));
        S_scaled_SD_precompute_col_ordered.block<1, 16>(i, 16) += m_SPK2_1_D0_Block.transpose().cwiseProduct(m_SD_precompute_col_ordered.block<1, 16>(i, 16));
        S_scaled_SD_precompute_col_ordered.block<1, 16>(i, 16) += m_SPK2_3_D0_Block.transpose().cwiseProduct(m_SD_precompute_col_ordered.block<1, 16>(i, 32));

        S_scaled_SD_precompute_col_ordered.block<1, 16>(i, 32) = m_SPK2_4_D0_Block.transpose().cwiseProduct(m_SD_precompute_col_ordered.block<1, 16>(i, 0));
        S_scaled_SD_precompute_col_ordered.block<1, 16>(i, 32) += m_SPK2_3_D0_Block.transpose().cwiseProduct(m_SD_precompute_col_ordered.block<1, 16>(i, 16));
        S_scaled_SD_precompute_col_ordered.block<1, 16>(i, 32) += m_SPK2_2_D0_Block.transpose().cwiseProduct(m_SD_precompute_col_ordered.block<1, 16>(i, 32));

        S_scaled_SD_precompute_col_ordered.block<1, 4>(i, 48) = m_Sdiag_0_Dv_Block.transpose().cwiseProduct(m_SD_precompute_col_ordered.block<1, 4>(i, 48));
        S_scaled_SD_precompute_col_ordered.block<1, 4>(i, 52) = m_Sdiag_1_Dv_Block.transpose().cwiseProduct(m_SD_precompute_col_ordered.block<1, 4>(i, 52));
        S_scaled_SD_precompute_col_ordered.block<1, 4>(i, 56) = m_Sdiag_2_Dv_Block.transpose().cwiseProduct(m_SD_precompute_col_ordered.block<1, 4>(i, 56));
    }

    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor, 9, 9> Jacobian_CompactPart;
    Jacobian_CompactPart.resize(9, 9);
    Jacobian_CompactPart = Mfactor * m_MassMatrix - Kfactor * m_SD_precompute_col_ordered * S_scaled_SD_precompute_col_ordered.transpose();

    //===========================================================================================

    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor, 9, 16> ChunkA1;
    ChunkA1.resize(9, 16);
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor, 9, 16> ChunkA2;
    ChunkA2.resize(9, 16);
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor, 9, 16> ChunkA3;
    ChunkA3.resize(9, 16);
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor, 9, 16> ChunkA4;
    ChunkA4.resize(9, 16);
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor, 9, 16> ChunkA5;
    ChunkA5.resize(9, 16);
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor, 9, 16> ChunkA6;
    ChunkA6.resize(9, 16);
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor, 9, 16> ChunkA7;
    ChunkA7.resize(9, 16);
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor, 9, 16> ChunkA8;
    ChunkA8.resize(9, 16);
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor, 9, 16> ChunkA9;
    ChunkA9.resize(9, 16);

    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor, 9, 16> ChunkB1;
    ChunkB1.resize(9, 16);
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor, 9, 16> ChunkB2;
    ChunkB2.resize(9, 16);
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor, 9, 16> ChunkB3;
    ChunkB3.resize(9, 16);
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor, 9, 16> ChunkB4;
    ChunkB4.resize(9, 16);
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor, 9, 16> ChunkB5;
    ChunkB5.resize(9, 16);
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor, 9, 16> ChunkB6;
    ChunkB6.resize(9, 16);
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor, 9, 16> ChunkB7;
    ChunkB7.resize(9, 16);
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor, 9, 16> ChunkB8;
    ChunkB8.resize(9, 16);
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor, 9, 16> ChunkB9;
    ChunkB9.resize(9, 16);

    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor, 9, 16> ChunkC1;
    ChunkC1.resize(9, 16);
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor, 9, 16> ChunkC2;
    ChunkC2.resize(9, 16);
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor, 9, 16> ChunkC3;
    ChunkC3.resize(9, 16);
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor, 9, 16> ChunkC4;
    ChunkC4.resize(9, 16);
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor, 9, 16> ChunkC5;
    ChunkC5.resize(9, 16);
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor, 9, 16> ChunkC6;
    ChunkC6.resize(9, 16);
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor, 9, 16> ChunkC7;
    ChunkC7.resize(9, 16);
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor, 9, 16> ChunkC8;
    ChunkC8.resize(9, 16);
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor, 9, 16> ChunkC9;
    ChunkC9.resize(9, 16);

    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor, 9, 4> ChunkDvA1;
    ChunkDvA1.resize(9, 4);
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor, 9, 4> ChunkDvA2;
    ChunkDvA2.resize(9, 4);
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor, 9, 4> ChunkDvA3;
    ChunkDvA3.resize(9, 4);

    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor, 9, 4> ChunkDvB4;
    ChunkDvB4.resize(9, 4);
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor, 9, 4> ChunkDvB5;
    ChunkDvB5.resize(9, 4);
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor, 9, 4> ChunkDvB6;
    ChunkDvB6.resize(9, 4);

    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor, 9, 4> ChunkDvC7;
    ChunkDvC7.resize(9, 4);
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor, 9, 4> ChunkDvC8;
    ChunkDvC8.resize(9, 4);
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor, 9, 4> ChunkDvC9;
    ChunkDvC9.resize(9, 4);

    for (auto i = 0; i < 9; i++) {
        ChunkA1.row(i) = m_SD_precompute_col_ordered.block<1, 16>(i, 0).cwiseProduct(m_F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(0, 0).transpose());
        ChunkA2.row(i) = m_SD_precompute_col_ordered.block<1, 16>(i, 0).cwiseProduct(m_F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(0, 1).transpose());
        ChunkA3.row(i) = m_SD_precompute_col_ordered.block<1, 16>(i, 0).cwiseProduct(m_F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(0, 2).transpose());
        ChunkA4.row(i) = m_SD_precompute_col_ordered.block<1, 16>(i, 16).cwiseProduct(m_F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(0, 0).transpose());
        ChunkA5.row(i) = m_SD_precompute_col_ordered.block<1, 16>(i, 16).cwiseProduct(m_F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(0, 1).transpose());
        ChunkA6.row(i) = m_SD_precompute_col_ordered.block<1, 16>(i, 16).cwiseProduct(m_F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(0, 2).transpose());
        ChunkA7.row(i) = m_SD_precompute_col_ordered.block<1, 16>(i, 32).cwiseProduct(m_F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(0, 0).transpose());
        ChunkA8.row(i) = m_SD_precompute_col_ordered.block<1, 16>(i, 32).cwiseProduct(m_F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(0, 1).transpose());
        ChunkA9.row(i) = m_SD_precompute_col_ordered.block<1, 16>(i, 32).cwiseProduct(m_F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(0, 2).transpose());

        ChunkB1.row(i) = m_SD_precompute_col_ordered.block<1, 16>(i, 0).cwiseProduct(m_F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(16, 0).transpose());
        ChunkB2.row(i) = m_SD_precompute_col_ordered.block<1, 16>(i, 0).cwiseProduct(m_F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(16, 1).transpose());
        ChunkB3.row(i) = m_SD_precompute_col_ordered.block<1, 16>(i, 0).cwiseProduct(m_F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(16, 2).transpose());
        ChunkB4.row(i) = m_SD_precompute_col_ordered.block<1, 16>(i, 16).cwiseProduct(m_F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(16, 0).transpose());
        ChunkB5.row(i) = m_SD_precompute_col_ordered.block<1, 16>(i, 16).cwiseProduct(m_F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(16, 1).transpose());
        ChunkB6.row(i) = m_SD_precompute_col_ordered.block<1, 16>(i, 16).cwiseProduct(m_F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(16, 2).transpose());
        ChunkB7.row(i) = m_SD_precompute_col_ordered.block<1, 16>(i, 32).cwiseProduct(m_F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(16, 0).transpose());
        ChunkB8.row(i) = m_SD_precompute_col_ordered.block<1, 16>(i, 32).cwiseProduct(m_F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(16, 1).transpose());
        ChunkB9.row(i) = m_SD_precompute_col_ordered.block<1, 16>(i, 32).cwiseProduct(m_F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(16, 2).transpose());

        ChunkC1.row(i) = m_SD_precompute_col_ordered.block<1, 16>(i, 0).cwiseProduct(m_F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(32, 0).transpose());
        ChunkC2.row(i) = m_SD_precompute_col_ordered.block<1, 16>(i, 0).cwiseProduct(m_F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(32, 1).transpose());
        ChunkC3.row(i) = m_SD_precompute_col_ordered.block<1, 16>(i, 0).cwiseProduct(m_F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(32, 2).transpose());
        ChunkC4.row(i) = m_SD_precompute_col_ordered.block<1, 16>(i, 16).cwiseProduct(m_F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(32, 0).transpose());
        ChunkC5.row(i) = m_SD_precompute_col_ordered.block<1, 16>(i, 16).cwiseProduct(m_F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(32, 1).transpose());
        ChunkC6.row(i) = m_SD_precompute_col_ordered.block<1, 16>(i, 16).cwiseProduct(m_F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(32, 2).transpose());
        ChunkC7.row(i) = m_SD_precompute_col_ordered.block<1, 16>(i, 32).cwiseProduct(m_F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(32, 0).transpose());
        ChunkC8.row(i) = m_SD_precompute_col_ordered.block<1, 16>(i, 32).cwiseProduct(m_F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(32, 1).transpose());
        ChunkC9.row(i) = m_SD_precompute_col_ordered.block<1, 16>(i, 32).cwiseProduct(m_F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(32, 2).transpose());

        ChunkDvA1.row(i) = m_SD_precompute_col_ordered.block<1, 4>(i, 48).cwiseProduct(m_F_Transpose_CombinedBlockDamping_col_ordered.block<4, 1>(48, 0).transpose());
        ChunkDvA2.row(i) = m_SD_precompute_col_ordered.block<1, 4>(i, 48).cwiseProduct(m_F_Transpose_CombinedBlockDamping_col_ordered.block<4, 1>(48, 1).transpose());
        ChunkDvA3.row(i) = m_SD_precompute_col_ordered.block<1, 4>(i, 48).cwiseProduct(m_F_Transpose_CombinedBlockDamping_col_ordered.block<4, 1>(48, 2).transpose());

        ChunkDvB4.row(i) = m_SD_precompute_col_ordered.block<1, 4>(i, 52).cwiseProduct(m_F_Transpose_CombinedBlockDamping_col_ordered.block<4, 1>(52, 0).transpose());
        ChunkDvB5.row(i) = m_SD_precompute_col_ordered.block<1, 4>(i, 52).cwiseProduct(m_F_Transpose_CombinedBlockDamping_col_ordered.block<4, 1>(52, 1).transpose());
        ChunkDvB6.row(i) = m_SD_precompute_col_ordered.block<1, 4>(i, 52).cwiseProduct(m_F_Transpose_CombinedBlockDamping_col_ordered.block<4, 1>(52, 2).transpose());

        ChunkDvC7.row(i) = m_SD_precompute_col_ordered.block<1, 4>(i, 56).cwiseProduct(m_F_Transpose_CombinedBlockDamping_col_ordered.block<4, 1>(56, 0).transpose());
        ChunkDvC8.row(i) = m_SD_precompute_col_ordered.block<1, 4>(i, 56).cwiseProduct(m_F_Transpose_CombinedBlockDamping_col_ordered.block<4, 1>(56, 1).transpose());
        ChunkDvC9.row(i) = m_SD_precompute_col_ordered.block<1, 4>(i, 56).cwiseProduct(m_F_Transpose_CombinedBlockDamping_col_ordered.block<4, 1>(56, 2).transpose());
    }
    

    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor, 9, 16> ChunkSA1;
    ChunkSA1.resize(9, 16);
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor, 9, 16> ChunkSA2;
    ChunkSA2.resize(9, 16);
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor, 9, 16> ChunkSA3;
    ChunkSA3.resize(9, 16);
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor, 9, 16> ChunkSA4;
    ChunkSA4.resize(9, 16);
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor, 9, 16> ChunkSA5;
    ChunkSA5.resize(9, 16);
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor, 9, 16> ChunkSA6;
    ChunkSA6.resize(9, 16);
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor, 9, 16> ChunkSA7;
    ChunkSA7.resize(9, 16);
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor, 9, 16> ChunkSA8;
    ChunkSA8.resize(9, 16);
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor, 9, 16> ChunkSA9;
    ChunkSA9.resize(9, 16);

    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor, 9, 16> ChunkSB1;
    ChunkSB1.resize(9, 16);
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor, 9, 16> ChunkSB2;
    ChunkSB2.resize(9, 16);
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor, 9, 16> ChunkSB3;
    ChunkSB3.resize(9, 16);
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor, 9, 16> ChunkSB4;
    ChunkSB4.resize(9, 16);
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor, 9, 16> ChunkSB5;
    ChunkSB5.resize(9, 16);
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor, 9, 16> ChunkSB6;
    ChunkSB6.resize(9, 16);
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor, 9, 16> ChunkSB7;
    ChunkSB7.resize(9, 16);
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor, 9, 16> ChunkSB8;
    ChunkSB8.resize(9, 16);
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor, 9, 16> ChunkSB9;
    ChunkSB9.resize(9, 16);

    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor, 9, 16> ChunkSC1;
    ChunkSC1.resize(9, 16);
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor, 9, 16> ChunkSC2;
    ChunkSC2.resize(9, 16);
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor, 9, 16> ChunkSC3;
    ChunkSC3.resize(9, 16);
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor, 9, 16> ChunkSC4;
    ChunkSC4.resize(9, 16);
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor, 9, 16> ChunkSC5;
    ChunkSC5.resize(9, 16);
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor, 9, 16> ChunkSC6;
    ChunkSC6.resize(9, 16);
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor, 9, 16> ChunkSC7;
    ChunkSC7.resize(9, 16);
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor, 9, 16> ChunkSC8;
    ChunkSC8.resize(9, 16);
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor, 9, 16> ChunkSC9;
    ChunkSC9.resize(9, 16);

    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor, 9, 4> ChunkSDvA1;
    ChunkSDvA1.resize(9, 4);
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor, 9, 4> ChunkSDvA2;
    ChunkSDvA2.resize(9, 4);
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor, 9, 4> ChunkSDvA3;
    ChunkSDvA3.resize(9, 4);

    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor, 9, 4> ChunkSDvB4;
    ChunkSDvB4.resize(9, 4);
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor, 9, 4> ChunkSDvB5;
    ChunkSDvB5.resize(9, 4);
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor, 9, 4> ChunkSDvB6;
    ChunkSDvB6.resize(9, 4);

    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor, 9, 4> ChunkSDvC7;
    ChunkSDvC7.resize(9, 4);
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor, 9, 4> ChunkSDvC8;
    ChunkSDvC8.resize(9, 4);
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor, 9, 4> ChunkSDvC9;
    ChunkSDvC9.resize(9, 4);

    for (auto i = 0; i < 9; i++) {
        ChunkSA1.row(i) = m_SD_precompute_col_ordered.block<1, 16>(i, 0).cwiseProduct(m_GQWeight_det_J_0xi_D0.transpose().cwiseProduct((Kfactor + m_Alpha * Rfactor)*m_F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(0, 0).transpose() + (m_Alpha*Kfactor)*m_F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(0, 3).transpose()));
        ChunkSA2.row(i) = m_SD_precompute_col_ordered.block<1, 16>(i, 0).cwiseProduct(m_GQWeight_det_J_0xi_D0.transpose().cwiseProduct((Kfactor + m_Alpha * Rfactor)*m_F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(0, 1).transpose() + (m_Alpha*Kfactor)*m_F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(0, 4).transpose()));
        ChunkSA3.row(i) = m_SD_precompute_col_ordered.block<1, 16>(i, 0).cwiseProduct(m_GQWeight_det_J_0xi_D0.transpose().cwiseProduct((Kfactor + m_Alpha * Rfactor)*m_F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(0, 2).transpose() + (m_Alpha*Kfactor)*m_F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(0, 5).transpose()));
        ChunkSA4.row(i) = m_SD_precompute_col_ordered.block<1, 16>(i, 16).cwiseProduct(m_GQWeight_det_J_0xi_D0.transpose().cwiseProduct((Kfactor + m_Alpha * Rfactor)*m_F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(0, 0).transpose() + (m_Alpha*Kfactor)*m_F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(0, 3).transpose()));
        ChunkSA5.row(i) = m_SD_precompute_col_ordered.block<1, 16>(i, 16).cwiseProduct(m_GQWeight_det_J_0xi_D0.transpose().cwiseProduct((Kfactor + m_Alpha * Rfactor)*m_F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(0, 1).transpose() + (m_Alpha*Kfactor)*m_F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(0, 4).transpose()));
        ChunkSA6.row(i) = m_SD_precompute_col_ordered.block<1, 16>(i, 16).cwiseProduct(m_GQWeight_det_J_0xi_D0.transpose().cwiseProduct((Kfactor + m_Alpha * Rfactor)*m_F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(0, 2).transpose() + (m_Alpha*Kfactor)*m_F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(0, 5).transpose()));
        ChunkSA7.row(i) = m_SD_precompute_col_ordered.block<1, 16>(i, 32).cwiseProduct(m_GQWeight_det_J_0xi_D0.transpose().cwiseProduct((Kfactor + m_Alpha * Rfactor)*m_F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(0, 0).transpose() + (m_Alpha*Kfactor)*m_F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(0, 3).transpose()));
        ChunkSA8.row(i) = m_SD_precompute_col_ordered.block<1, 16>(i, 32).cwiseProduct(m_GQWeight_det_J_0xi_D0.transpose().cwiseProduct((Kfactor + m_Alpha * Rfactor)*m_F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(0, 1).transpose() + (m_Alpha*Kfactor)*m_F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(0, 4).transpose()));
        ChunkSA9.row(i) = m_SD_precompute_col_ordered.block<1, 16>(i, 32).cwiseProduct(m_GQWeight_det_J_0xi_D0.transpose().cwiseProduct((Kfactor + m_Alpha * Rfactor)*m_F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(0, 2).transpose() + (m_Alpha*Kfactor)*m_F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(0, 5).transpose()));

        ChunkSB1.row(i) = m_SD_precompute_col_ordered.block<1, 16>(i, 0).cwiseProduct(m_GQWeight_det_J_0xi_D0.transpose().cwiseProduct((Kfactor + m_Alpha * Rfactor)*m_F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(16, 0).transpose() + (m_Alpha*Kfactor)*m_F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(16, 3).transpose()));
        ChunkSB2.row(i) = m_SD_precompute_col_ordered.block<1, 16>(i, 0).cwiseProduct(m_GQWeight_det_J_0xi_D0.transpose().cwiseProduct((Kfactor + m_Alpha * Rfactor)*m_F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(16, 1).transpose() + (m_Alpha*Kfactor)*m_F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(16, 4).transpose()));
        ChunkSB3.row(i) = m_SD_precompute_col_ordered.block<1, 16>(i, 0).cwiseProduct(m_GQWeight_det_J_0xi_D0.transpose().cwiseProduct((Kfactor + m_Alpha * Rfactor)*m_F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(16, 2).transpose() + (m_Alpha*Kfactor)*m_F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(16, 5).transpose()));
        ChunkSB4.row(i) = m_SD_precompute_col_ordered.block<1, 16>(i, 16).cwiseProduct(m_GQWeight_det_J_0xi_D0.transpose().cwiseProduct((Kfactor + m_Alpha * Rfactor)*m_F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(16, 0).transpose() + (m_Alpha*Kfactor)*m_F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(16, 3).transpose()));
        ChunkSB5.row(i) = m_SD_precompute_col_ordered.block<1, 16>(i, 16).cwiseProduct(m_GQWeight_det_J_0xi_D0.transpose().cwiseProduct((Kfactor + m_Alpha * Rfactor)*m_F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(16, 1).transpose() + (m_Alpha*Kfactor)*m_F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(16, 4).transpose()));
        ChunkSB6.row(i) = m_SD_precompute_col_ordered.block<1, 16>(i, 16).cwiseProduct(m_GQWeight_det_J_0xi_D0.transpose().cwiseProduct((Kfactor + m_Alpha * Rfactor)*m_F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(16, 2).transpose() + (m_Alpha*Kfactor)*m_F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(16, 5).transpose()));
        ChunkSB7.row(i) = m_SD_precompute_col_ordered.block<1, 16>(i, 32).cwiseProduct(m_GQWeight_det_J_0xi_D0.transpose().cwiseProduct((Kfactor + m_Alpha * Rfactor)*m_F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(16, 0).transpose() + (m_Alpha*Kfactor)*m_F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(16, 3).transpose()));
        ChunkSB8.row(i) = m_SD_precompute_col_ordered.block<1, 16>(i, 32).cwiseProduct(m_GQWeight_det_J_0xi_D0.transpose().cwiseProduct((Kfactor + m_Alpha * Rfactor)*m_F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(16, 1).transpose() + (m_Alpha*Kfactor)*m_F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(16, 4).transpose()));
        ChunkSB9.row(i) = m_SD_precompute_col_ordered.block<1, 16>(i, 32).cwiseProduct(m_GQWeight_det_J_0xi_D0.transpose().cwiseProduct((Kfactor + m_Alpha * Rfactor)*m_F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(16, 2).transpose() + (m_Alpha*Kfactor)*m_F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(16, 5).transpose()));

        ChunkSC1.row(i) = m_SD_precompute_col_ordered.block<1, 16>(i, 0).cwiseProduct(m_GQWeight_det_J_0xi_D0.transpose().cwiseProduct((Kfactor + m_Alpha * Rfactor)*m_F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(32, 0).transpose() + (m_Alpha*Kfactor)*m_F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(32, 3).transpose()));
        ChunkSC2.row(i) = m_SD_precompute_col_ordered.block<1, 16>(i, 0).cwiseProduct(m_GQWeight_det_J_0xi_D0.transpose().cwiseProduct((Kfactor + m_Alpha * Rfactor)*m_F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(32, 1).transpose() + (m_Alpha*Kfactor)*m_F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(32, 4).transpose()));
        ChunkSC3.row(i) = m_SD_precompute_col_ordered.block<1, 16>(i, 0).cwiseProduct(m_GQWeight_det_J_0xi_D0.transpose().cwiseProduct((Kfactor + m_Alpha * Rfactor)*m_F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(32, 2).transpose() + (m_Alpha*Kfactor)*m_F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(32, 5).transpose()));
        ChunkSC4.row(i) = m_SD_precompute_col_ordered.block<1, 16>(i, 16).cwiseProduct(m_GQWeight_det_J_0xi_D0.transpose().cwiseProduct((Kfactor + m_Alpha * Rfactor)*m_F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(32, 0).transpose() + (m_Alpha*Kfactor)*m_F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(32, 3).transpose()));
        ChunkSC5.row(i) = m_SD_precompute_col_ordered.block<1, 16>(i, 16).cwiseProduct(m_GQWeight_det_J_0xi_D0.transpose().cwiseProduct((Kfactor + m_Alpha * Rfactor)*m_F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(32, 1).transpose() + (m_Alpha*Kfactor)*m_F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(32, 4).transpose()));
        ChunkSC6.row(i) = m_SD_precompute_col_ordered.block<1, 16>(i, 16).cwiseProduct(m_GQWeight_det_J_0xi_D0.transpose().cwiseProduct((Kfactor + m_Alpha * Rfactor)*m_F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(32, 2).transpose() + (m_Alpha*Kfactor)*m_F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(32, 5).transpose()));
        ChunkSC7.row(i) = m_SD_precompute_col_ordered.block<1, 16>(i, 32).cwiseProduct(m_GQWeight_det_J_0xi_D0.transpose().cwiseProduct((Kfactor + m_Alpha * Rfactor)*m_F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(32, 0).transpose() + (m_Alpha*Kfactor)*m_F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(32, 3).transpose()));
        ChunkSC8.row(i) = m_SD_precompute_col_ordered.block<1, 16>(i, 32).cwiseProduct(m_GQWeight_det_J_0xi_D0.transpose().cwiseProduct((Kfactor + m_Alpha * Rfactor)*m_F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(32, 1).transpose() + (m_Alpha*Kfactor)*m_F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(32, 4).transpose()));
        ChunkSC9.row(i) = m_SD_precompute_col_ordered.block<1, 16>(i, 32).cwiseProduct(m_GQWeight_det_J_0xi_D0.transpose().cwiseProduct((Kfactor + m_Alpha * Rfactor)*m_F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(32, 2).transpose() + (m_Alpha*Kfactor)*m_F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(32, 5).transpose()));

        ChunkSDvA1.row(i) = m_SD_precompute_col_ordered.block<1, 4>(i, 48).cwiseProduct(m_GQWeight_det_J_0xi_Dv.transpose().cwiseProduct((Kfactor + m_Alpha * Rfactor)*m_F_Transpose_CombinedBlockDamping_col_ordered.block<4, 1>(48, 0).transpose() + (m_Alpha*Kfactor)*m_F_Transpose_CombinedBlockDamping_col_ordered.block<4, 1>(48, 3).transpose()));
        ChunkSDvA2.row(i) = m_SD_precompute_col_ordered.block<1, 4>(i, 48).cwiseProduct(m_GQWeight_det_J_0xi_Dv.transpose().cwiseProduct((Kfactor + m_Alpha * Rfactor)*m_F_Transpose_CombinedBlockDamping_col_ordered.block<4, 1>(48, 1).transpose() + (m_Alpha*Kfactor)*m_F_Transpose_CombinedBlockDamping_col_ordered.block<4, 1>(48, 4).transpose()));
        ChunkSDvA3.row(i) = m_SD_precompute_col_ordered.block<1, 4>(i, 48).cwiseProduct(m_GQWeight_det_J_0xi_Dv.transpose().cwiseProduct((Kfactor + m_Alpha * Rfactor)*m_F_Transpose_CombinedBlockDamping_col_ordered.block<4, 1>(48, 2).transpose() + (m_Alpha*Kfactor)*m_F_Transpose_CombinedBlockDamping_col_ordered.block<4, 1>(48, 5).transpose()));

        ChunkSDvB4.row(i) = m_SD_precompute_col_ordered.block<1, 4>(i, 52).cwiseProduct(m_GQWeight_det_J_0xi_Dv.transpose().cwiseProduct((Kfactor + m_Alpha * Rfactor)*m_F_Transpose_CombinedBlockDamping_col_ordered.block<4, 1>(52, 0).transpose() + (m_Alpha*Kfactor)*m_F_Transpose_CombinedBlockDamping_col_ordered.block<4, 1>(52, 3).transpose()));
        ChunkSDvB5.row(i) = m_SD_precompute_col_ordered.block<1, 4>(i, 52).cwiseProduct(m_GQWeight_det_J_0xi_Dv.transpose().cwiseProduct((Kfactor + m_Alpha * Rfactor)*m_F_Transpose_CombinedBlockDamping_col_ordered.block<4, 1>(52, 1).transpose() + (m_Alpha*Kfactor)*m_F_Transpose_CombinedBlockDamping_col_ordered.block<4, 1>(52, 4).transpose()));
        ChunkSDvB6.row(i) = m_SD_precompute_col_ordered.block<1, 4>(i, 52).cwiseProduct(m_GQWeight_det_J_0xi_Dv.transpose().cwiseProduct((Kfactor + m_Alpha * Rfactor)*m_F_Transpose_CombinedBlockDamping_col_ordered.block<4, 1>(52, 2).transpose() + (m_Alpha*Kfactor)*m_F_Transpose_CombinedBlockDamping_col_ordered.block<4, 1>(52, 5).transpose()));

        ChunkSDvC7.row(i) = m_SD_precompute_col_ordered.block<1, 4>(i, 56).cwiseProduct(m_GQWeight_det_J_0xi_Dv.transpose().cwiseProduct((Kfactor + m_Alpha * Rfactor)*m_F_Transpose_CombinedBlockDamping_col_ordered.block<4, 1>(56, 0).transpose() + (m_Alpha*Kfactor)*m_F_Transpose_CombinedBlockDamping_col_ordered.block<4, 1>(56, 3).transpose()));
        ChunkSDvC8.row(i) = m_SD_precompute_col_ordered.block<1, 4>(i, 56).cwiseProduct(m_GQWeight_det_J_0xi_Dv.transpose().cwiseProduct((Kfactor + m_Alpha * Rfactor)*m_F_Transpose_CombinedBlockDamping_col_ordered.block<4, 1>(56, 1).transpose() + (m_Alpha*Kfactor)*m_F_Transpose_CombinedBlockDamping_col_ordered.block<4, 1>(56, 4).transpose()));
        ChunkSDvC9.row(i) = m_SD_precompute_col_ordered.block<1, 4>(i, 56).cwiseProduct(m_GQWeight_det_J_0xi_Dv.transpose().cwiseProduct((Kfactor + m_Alpha * Rfactor)*m_F_Transpose_CombinedBlockDamping_col_ordered.block<4, 1>(56, 2).transpose() + (m_Alpha*Kfactor)*m_F_Transpose_CombinedBlockDamping_col_ordered.block<4, 1>(56, 5).transpose()));
    }


    //ChMatrixNMc<double, 108, 9> partial_epsilon_partial_e_Block11;
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor, 108, 9> partial_epsilon_partial_e_Block11;
    partial_epsilon_partial_e_Block11.resize(108, 9);
    partial_epsilon_partial_e_Block11.block<16, 9>(0, 0) = ChunkA1.transpose();
    partial_epsilon_partial_e_Block11.block<16, 9>(16, 0) = ChunkB4.transpose();
    partial_epsilon_partial_e_Block11.block<16, 9>(32, 0) = ChunkC7.transpose();
    partial_epsilon_partial_e_Block11.block<16, 9>(48, 0) = ChunkB7.transpose() + ChunkC4.transpose();
    partial_epsilon_partial_e_Block11.block<16, 9>(64, 0) = ChunkA7.transpose() + ChunkC1.transpose();
    partial_epsilon_partial_e_Block11.block<16, 9>(80, 0) = ChunkA4.transpose() + ChunkB1.transpose();
    partial_epsilon_partial_e_Block11.block<4, 9>(96, 0) = ChunkDvA1.transpose();
    partial_epsilon_partial_e_Block11.block<4, 9>(100, 0) = ChunkDvB4.transpose();
    partial_epsilon_partial_e_Block11.block<4, 9>(104, 0) = ChunkDvC7.transpose();

    //ChMatrixNMc<double, 108, 9> partial_epsilon_partial_e_Block22;
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor, 108, 9> partial_epsilon_partial_e_Block22;
    partial_epsilon_partial_e_Block22.resize(108, 9);
    partial_epsilon_partial_e_Block22.block<16, 9>(0, 0) = ChunkA2.transpose();
    partial_epsilon_partial_e_Block22.block<16, 9>(16, 0) = ChunkB5.transpose();
    partial_epsilon_partial_e_Block22.block<16, 9>(32, 0) = ChunkC8.transpose();
    partial_epsilon_partial_e_Block22.block<16, 9>(48, 0) = ChunkB8.transpose() + ChunkC5.transpose();
    partial_epsilon_partial_e_Block22.block<16, 9>(64, 0) = ChunkA8.transpose() + ChunkC2.transpose();
    partial_epsilon_partial_e_Block22.block<16, 9>(80, 0) = ChunkA5.transpose() + ChunkB2.transpose();
    partial_epsilon_partial_e_Block22.block<4, 9>(96, 0) = ChunkDvA2.transpose();
    partial_epsilon_partial_e_Block22.block<4, 9>(100, 0) = ChunkDvB5.transpose();
    partial_epsilon_partial_e_Block22.block<4, 9>(104, 0) = ChunkDvC8.transpose();

    //ChMatrixNMc<double, 108, 9> partial_epsilon_partial_e_Block33;
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor, 108, 9> partial_epsilon_partial_e_Block33;
    partial_epsilon_partial_e_Block33.resize(108, 9);
    partial_epsilon_partial_e_Block33.block<16, 9>(0, 0) = ChunkA3.transpose();
    partial_epsilon_partial_e_Block33.block<16, 9>(16, 0) = ChunkB6.transpose();
    partial_epsilon_partial_e_Block33.block<16, 9>(32, 0) = ChunkC9.transpose();
    partial_epsilon_partial_e_Block33.block<16, 9>(48, 0) = ChunkB9.transpose() + ChunkC6.transpose();
    partial_epsilon_partial_e_Block33.block<16, 9>(64, 0) = ChunkA9.transpose() + ChunkC3.transpose();
    partial_epsilon_partial_e_Block33.block<16, 9>(80, 0) = ChunkA6.transpose() + ChunkB3.transpose();
    partial_epsilon_partial_e_Block33.block<4, 9>(96, 0) = ChunkDvA3.transpose();
    partial_epsilon_partial_e_Block33.block<4, 9>(100, 0) = ChunkDvB6.transpose();
    partial_epsilon_partial_e_Block33.block<4, 9>(104, 0) = ChunkDvC9.transpose();


    ChVectorN<double, 6> D0 = GetMaterial()->Get_D0();
    ChMatrix33<double> Dv = GetMaterial()->Get_Dv();

    //ChMatrixNMc<double, 108, 9> Scaled_Combined_partial_epsilon_partial_e_Block11;
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor, 108, 9> Scaled_Combined_partial_epsilon_partial_e_Block11;
    Scaled_Combined_partial_epsilon_partial_e_Block11.resize(108, 9);
    Scaled_Combined_partial_epsilon_partial_e_Block11.block<16, 9>(0, 0) = D0(0)*ChunkSA1.transpose();
    Scaled_Combined_partial_epsilon_partial_e_Block11.block<16, 9>(16, 0) = D0(1)*ChunkSB4.transpose();
    Scaled_Combined_partial_epsilon_partial_e_Block11.block<16, 9>(32, 0) = D0(2)*ChunkSC7.transpose();
    Scaled_Combined_partial_epsilon_partial_e_Block11.block<16, 9>(48, 0) = D0(3)*(ChunkSB7.transpose() + ChunkSC4.transpose());
    Scaled_Combined_partial_epsilon_partial_e_Block11.block<16, 9>(64, 0) = D0(4)*(ChunkSA7.transpose() + ChunkSC1.transpose());
    Scaled_Combined_partial_epsilon_partial_e_Block11.block<16, 9>(80, 0) = D0(5)*(ChunkSA4.transpose() + ChunkSB1.transpose());
    Scaled_Combined_partial_epsilon_partial_e_Block11.block<4, 9>(96, 0) = Dv(0, 0)*ChunkSDvA1.transpose() + Dv(0, 1)*ChunkSDvB4.transpose() + Dv(0, 2)*ChunkSDvC7.transpose();
    Scaled_Combined_partial_epsilon_partial_e_Block11.block<4, 9>(100, 0) = Dv(1, 0)*ChunkSDvA1.transpose() + Dv(1, 1)*ChunkSDvB4.transpose() + Dv(1, 2)*ChunkSDvC7.transpose();
    Scaled_Combined_partial_epsilon_partial_e_Block11.block<4, 9>(104, 0) = Dv(2, 0)*ChunkSDvA1.transpose() + Dv(2, 1)*ChunkSDvB4.transpose() + Dv(2, 2)*ChunkSDvC7.transpose();

    //ChMatrixNMc<double, 108, 9> Scaled_Combined_partial_epsilon_partial_e_Block22;
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor, 108, 9> Scaled_Combined_partial_epsilon_partial_e_Block22;
    Scaled_Combined_partial_epsilon_partial_e_Block22.resize(108, 9);
    Scaled_Combined_partial_epsilon_partial_e_Block22.block<16, 9>(0, 0) = D0(0)*ChunkSA2.transpose();
    Scaled_Combined_partial_epsilon_partial_e_Block22.block<16, 9>(16, 0) = D0(1)*ChunkSB5.transpose();
    Scaled_Combined_partial_epsilon_partial_e_Block22.block<16, 9>(32, 0) = D0(2)*ChunkSC8.transpose();
    Scaled_Combined_partial_epsilon_partial_e_Block22.block<16, 9>(48, 0) = D0(3)*(ChunkSB8.transpose() + ChunkSC5.transpose());
    Scaled_Combined_partial_epsilon_partial_e_Block22.block<16, 9>(64, 0) = D0(4)*(ChunkSA8.transpose() + ChunkSC2.transpose());
    Scaled_Combined_partial_epsilon_partial_e_Block22.block<16, 9>(80, 0) = D0(5)*(ChunkSA5.transpose() + ChunkSB2.transpose());
    Scaled_Combined_partial_epsilon_partial_e_Block22.block<4, 9>(96, 0) = Dv(0, 0)*ChunkSDvA2.transpose() + Dv(0, 1)*ChunkSDvB5.transpose() + Dv(0, 2)*ChunkSDvC8.transpose();
    Scaled_Combined_partial_epsilon_partial_e_Block22.block<4, 9>(100, 0) = Dv(1, 0)*ChunkSDvA2.transpose() + Dv(1, 1)*ChunkSDvB5.transpose() + Dv(1, 2)*ChunkSDvC8.transpose();
    Scaled_Combined_partial_epsilon_partial_e_Block22.block<4, 9>(104, 0) = Dv(2, 0)*ChunkSDvA2.transpose() + Dv(2, 1)*ChunkSDvB5.transpose() + Dv(2, 2)*ChunkSDvC8.transpose();

    //ChMatrixNMc<double, 108, 9> Scaled_Combined_partial_epsilon_partial_e_Block33;
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor, 108, 9> Scaled_Combined_partial_epsilon_partial_e_Block33;
    Scaled_Combined_partial_epsilon_partial_e_Block33.resize(108, 9);
    Scaled_Combined_partial_epsilon_partial_e_Block33.block<16, 9>(0, 0) = D0(0)*ChunkSA3.transpose();
    Scaled_Combined_partial_epsilon_partial_e_Block33.block<16, 9>(16, 0) = D0(1)*ChunkSB6.transpose();
    Scaled_Combined_partial_epsilon_partial_e_Block33.block<16, 9>(32, 0) = D0(2)*ChunkSC9.transpose();
    Scaled_Combined_partial_epsilon_partial_e_Block33.block<16, 9>(48, 0) = D0(3)*ChunkSB9.transpose() + D0(3)*ChunkSC6.transpose();
    Scaled_Combined_partial_epsilon_partial_e_Block33.block<16, 9>(64, 0) = D0(4)*ChunkSA9.transpose() + D0(4)*ChunkSC3.transpose();
    Scaled_Combined_partial_epsilon_partial_e_Block33.block<16, 9>(80, 0) = D0(5)*ChunkSA6.transpose() + D0(5)*ChunkSB3.transpose();
    Scaled_Combined_partial_epsilon_partial_e_Block33.block<4, 9>(96, 0) = Dv(0, 0)*ChunkSDvA3.transpose() + Dv(0, 1)*ChunkSDvB6.transpose() + Dv(0, 2)*ChunkSDvC9.transpose();
    Scaled_Combined_partial_epsilon_partial_e_Block33.block<4, 9>(100, 0) = Dv(1, 0)*ChunkSDvA3.transpose() + Dv(1, 1)*ChunkSDvB6.transpose() + Dv(1, 2)*ChunkSDvC9.transpose();
    Scaled_Combined_partial_epsilon_partial_e_Block33.block<4, 9>(104, 0) = Dv(2, 0)*ChunkSDvA3.transpose() + Dv(2, 1)*ChunkSDvB6.transpose() + Dv(2, 2)*ChunkSDvC9.transpose();


    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor, 9, 9> Block11;
    Block11.resize(9, 9);
    Block11 = partial_epsilon_partial_e_Block11.transpose()*Scaled_Combined_partial_epsilon_partial_e_Block11;
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor, 9, 9> Block12;
    Block12.resize(9, 9);
    Block12 = partial_epsilon_partial_e_Block11.transpose()*Scaled_Combined_partial_epsilon_partial_e_Block22;
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor, 9, 9> Block13;
    Block13.resize(9, 9);
    Block13 = partial_epsilon_partial_e_Block11.transpose()*Scaled_Combined_partial_epsilon_partial_e_Block33;

    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor, 9, 9> Block21;
    Block21.resize(9, 9);
    Block21 = partial_epsilon_partial_e_Block22.transpose()*Scaled_Combined_partial_epsilon_partial_e_Block11;
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor, 9, 9> Block22;
    Block22.resize(9, 9);
    Block22 = partial_epsilon_partial_e_Block22.transpose()*Scaled_Combined_partial_epsilon_partial_e_Block22;
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor, 9, 9> Block23;
    Block23.resize(9, 9);
    Block23 = partial_epsilon_partial_e_Block22.transpose()*Scaled_Combined_partial_epsilon_partial_e_Block33;

    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor, 9, 9> Block31;
    Block31.resize(9, 9);
    Block31 = partial_epsilon_partial_e_Block33.transpose()*Scaled_Combined_partial_epsilon_partial_e_Block11;
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor, 9, 9> Block32;
    Block32.resize(9, 9);
    Block32 = partial_epsilon_partial_e_Block33.transpose()*Scaled_Combined_partial_epsilon_partial_e_Block22;
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor, 9, 9> Block33;
    Block33.resize(9, 9);
    Block33 = partial_epsilon_partial_e_Block33.transpose()*Scaled_Combined_partial_epsilon_partial_e_Block33;


    for (unsigned int i = 0; i < 9; i++) {
        for (unsigned int j = 0; j < 9; j++) {
            H(3 * i, 3 * j) = -Block11(i, j) + Jacobian_CompactPart(i, j);
            H(3 * i + 1, 3 * j) = -Block21(i, j);
            H(3 * i + 2, 3 * j) = -Block31(i, j);

            H(3 * i, 3 * j + 1) = -Block12(i, j);
            H(3 * i + 1, 3 * j + 1) = -Block22(i, j) + Jacobian_CompactPart(i, j);
            H(3 * i + 2, 3 * j + 1) = -Block32(i, j);

            H(3 * i, 3 * j + 2) = -Block13(i, j);
            H(3 * i + 1, 3 * j + 2) = -Block23(i, j);
            H(3 * i + 2, 3 * j + 2) = -Block33(i, j) + Jacobian_CompactPart(i, j);
        }
    }
}

//Calculate the calculate the Jacobian of the internal force integrand without damping included
void ChElementBeamANCF_MT35::ComputeInternalJacobianNoDamping(ChMatrixRef H, double Kfactor, double Rfactor, double Mfactor) {

    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor, 9, 60> S_scaled_SD_precompute_col_ordered;
    S_scaled_SD_precompute_col_ordered.resize(9, 60);

    for (auto i = 0; i < 9; i++) {
        S_scaled_SD_precompute_col_ordered.block<1, 16>(i, 0) = m_SPK2_0_D0_Block.transpose().cwiseProduct(m_SD_precompute_col_ordered.block<1, 16>(i, 0));
        S_scaled_SD_precompute_col_ordered.block<1, 16>(i, 0) += m_SPK2_5_D0_Block.transpose().cwiseProduct(m_SD_precompute_col_ordered.block<1, 16>(i, 16));
        S_scaled_SD_precompute_col_ordered.block<1, 16>(i, 0) += m_SPK2_4_D0_Block.transpose().cwiseProduct(m_SD_precompute_col_ordered.block<1, 16>(i, 32));

        S_scaled_SD_precompute_col_ordered.block<1, 16>(i, 16) = m_SPK2_5_D0_Block.transpose().cwiseProduct(m_SD_precompute_col_ordered.block<1, 16>(i, 0));
        S_scaled_SD_precompute_col_ordered.block<1, 16>(i, 16) += m_SPK2_1_D0_Block.transpose().cwiseProduct(m_SD_precompute_col_ordered.block<1, 16>(i, 16));
        S_scaled_SD_precompute_col_ordered.block<1, 16>(i, 16) += m_SPK2_3_D0_Block.transpose().cwiseProduct(m_SD_precompute_col_ordered.block<1, 16>(i, 32));

        S_scaled_SD_precompute_col_ordered.block<1, 16>(i, 32) = m_SPK2_4_D0_Block.transpose().cwiseProduct(m_SD_precompute_col_ordered.block<1, 16>(i, 0));
        S_scaled_SD_precompute_col_ordered.block<1, 16>(i, 32) += m_SPK2_3_D0_Block.transpose().cwiseProduct(m_SD_precompute_col_ordered.block<1, 16>(i, 16));
        S_scaled_SD_precompute_col_ordered.block<1, 16>(i, 32) += m_SPK2_2_D0_Block.transpose().cwiseProduct(m_SD_precompute_col_ordered.block<1, 16>(i, 32));

        S_scaled_SD_precompute_col_ordered.block<1, 4>(i, 48) = m_Sdiag_0_Dv_Block.transpose().cwiseProduct(m_SD_precompute_col_ordered.block<1, 4>(i, 48));
        S_scaled_SD_precompute_col_ordered.block<1, 4>(i, 52) = m_Sdiag_1_Dv_Block.transpose().cwiseProduct(m_SD_precompute_col_ordered.block<1, 4>(i, 52));
        S_scaled_SD_precompute_col_ordered.block<1, 4>(i, 56) = m_Sdiag_2_Dv_Block.transpose().cwiseProduct(m_SD_precompute_col_ordered.block<1, 4>(i, 56));
    }

    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor, 9, 9> Jacobian_CompactPart;
    Jacobian_CompactPart.resize(9, 9);
    Jacobian_CompactPart = Mfactor * m_MassMatrix - Kfactor * m_SD_precompute_col_ordered * S_scaled_SD_precompute_col_ordered.transpose();

    //===========================================================================================

    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor, 9, 16> ChunkA1;
    ChunkA1.resize(9, 16);
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor, 9, 16> ChunkA2;
    ChunkA2.resize(9, 16);
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor, 9, 16> ChunkA3;
    ChunkA3.resize(9, 16);
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor, 9, 16> ChunkA4;
    ChunkA4.resize(9, 16);
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor, 9, 16> ChunkA5;
    ChunkA5.resize(9, 16);
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor, 9, 16> ChunkA6;
    ChunkA6.resize(9, 16);
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor, 9, 16> ChunkA7;
    ChunkA7.resize(9, 16);
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor, 9, 16> ChunkA8;
    ChunkA8.resize(9, 16);
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor, 9, 16> ChunkA9;
    ChunkA9.resize(9, 16);

    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor, 9, 16> ChunkB1;
    ChunkB1.resize(9, 16);
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor, 9, 16> ChunkB2;
    ChunkB2.resize(9, 16);
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor, 9, 16> ChunkB3;
    ChunkB3.resize(9, 16);
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor, 9, 16> ChunkB4;
    ChunkB4.resize(9, 16);
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor, 9, 16> ChunkB5;
    ChunkB5.resize(9, 16);
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor, 9, 16> ChunkB6;
    ChunkB6.resize(9, 16);
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor, 9, 16> ChunkB7;
    ChunkB7.resize(9, 16);
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor, 9, 16> ChunkB8;
    ChunkB8.resize(9, 16);
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor, 9, 16> ChunkB9;
    ChunkB9.resize(9, 16);

    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor, 9, 16> ChunkC1;
    ChunkC1.resize(9, 16);
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor, 9, 16> ChunkC2;
    ChunkC2.resize(9, 16);
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor, 9, 16> ChunkC3;
    ChunkC3.resize(9, 16);
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor, 9, 16> ChunkC4;
    ChunkC4.resize(9, 16);
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor, 9, 16> ChunkC5;
    ChunkC5.resize(9, 16);
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor, 9, 16> ChunkC6;
    ChunkC6.resize(9, 16);
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor, 9, 16> ChunkC7;
    ChunkC7.resize(9, 16);
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor, 9, 16> ChunkC8;
    ChunkC8.resize(9, 16);
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor, 9, 16> ChunkC9;
    ChunkC9.resize(9, 16);

    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor, 9, 4> ChunkDvA1;
    ChunkDvA1.resize(9, 4);
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor, 9, 4> ChunkDvA2;
    ChunkDvA2.resize(9, 4);
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor, 9, 4> ChunkDvA3;
    ChunkDvA3.resize(9, 4);

    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor, 9, 4> ChunkDvB4;
    ChunkDvB4.resize(9, 4);
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor, 9, 4> ChunkDvB5;
    ChunkDvB5.resize(9, 4);
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor, 9, 4> ChunkDvB6;
    ChunkDvB6.resize(9, 4);

    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor, 9, 4> ChunkDvC7;
    ChunkDvC7.resize(9, 4);
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor, 9, 4> ChunkDvC8;
    ChunkDvC8.resize(9, 4);
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor, 9, 4> ChunkDvC9;
    ChunkDvC9.resize(9, 4);

    for (auto i = 0; i < 9; i++) {
        ChunkA1.row(i) = m_SD_precompute_col_ordered.block<1, 16>(i, 0).cwiseProduct(m_F_Transpose_CombinedBlock_col_ordered.block<16, 1>(0, 0).transpose());
        ChunkA2.row(i) = m_SD_precompute_col_ordered.block<1, 16>(i, 0).cwiseProduct(m_F_Transpose_CombinedBlock_col_ordered.block<16, 1>(0, 1).transpose());
        ChunkA3.row(i) = m_SD_precompute_col_ordered.block<1, 16>(i, 0).cwiseProduct(m_F_Transpose_CombinedBlock_col_ordered.block<16, 1>(0, 2).transpose());
        ChunkA4.row(i) = m_SD_precompute_col_ordered.block<1, 16>(i, 16).cwiseProduct(m_F_Transpose_CombinedBlock_col_ordered.block<16, 1>(0, 0).transpose());
        ChunkA5.row(i) = m_SD_precompute_col_ordered.block<1, 16>(i, 16).cwiseProduct(m_F_Transpose_CombinedBlock_col_ordered.block<16, 1>(0, 1).transpose());
        ChunkA6.row(i) = m_SD_precompute_col_ordered.block<1, 16>(i, 16).cwiseProduct(m_F_Transpose_CombinedBlock_col_ordered.block<16, 1>(0, 2).transpose());
        ChunkA7.row(i) = m_SD_precompute_col_ordered.block<1, 16>(i, 32).cwiseProduct(m_F_Transpose_CombinedBlock_col_ordered.block<16, 1>(0, 0).transpose());
        ChunkA8.row(i) = m_SD_precompute_col_ordered.block<1, 16>(i, 32).cwiseProduct(m_F_Transpose_CombinedBlock_col_ordered.block<16, 1>(0, 1).transpose());
        ChunkA9.row(i) = m_SD_precompute_col_ordered.block<1, 16>(i, 32).cwiseProduct(m_F_Transpose_CombinedBlock_col_ordered.block<16, 1>(0, 2).transpose());

        ChunkB1.row(i) = m_SD_precompute_col_ordered.block<1, 16>(i, 0).cwiseProduct(m_F_Transpose_CombinedBlock_col_ordered.block<16, 1>(16, 0).transpose());
        ChunkB2.row(i) = m_SD_precompute_col_ordered.block<1, 16>(i, 0).cwiseProduct(m_F_Transpose_CombinedBlock_col_ordered.block<16, 1>(16, 1).transpose());
        ChunkB3.row(i) = m_SD_precompute_col_ordered.block<1, 16>(i, 0).cwiseProduct(m_F_Transpose_CombinedBlock_col_ordered.block<16, 1>(16, 2).transpose());
        ChunkB4.row(i) = m_SD_precompute_col_ordered.block<1, 16>(i, 16).cwiseProduct(m_F_Transpose_CombinedBlock_col_ordered.block<16, 1>(16, 0).transpose());
        ChunkB5.row(i) = m_SD_precompute_col_ordered.block<1, 16>(i, 16).cwiseProduct(m_F_Transpose_CombinedBlock_col_ordered.block<16, 1>(16, 1).transpose());
        ChunkB6.row(i) = m_SD_precompute_col_ordered.block<1, 16>(i, 16).cwiseProduct(m_F_Transpose_CombinedBlock_col_ordered.block<16, 1>(16, 2).transpose());
        ChunkB7.row(i) = m_SD_precompute_col_ordered.block<1, 16>(i, 32).cwiseProduct(m_F_Transpose_CombinedBlock_col_ordered.block<16, 1>(16, 0).transpose());
        ChunkB8.row(i) = m_SD_precompute_col_ordered.block<1, 16>(i, 32).cwiseProduct(m_F_Transpose_CombinedBlock_col_ordered.block<16, 1>(16, 1).transpose());
        ChunkB9.row(i) = m_SD_precompute_col_ordered.block<1, 16>(i, 32).cwiseProduct(m_F_Transpose_CombinedBlock_col_ordered.block<16, 1>(16, 2).transpose());

        ChunkC1.row(i) = m_SD_precompute_col_ordered.block<1, 16>(i, 0).cwiseProduct(m_F_Transpose_CombinedBlock_col_ordered.block<16, 1>(32, 0).transpose());
        ChunkC2.row(i) = m_SD_precompute_col_ordered.block<1, 16>(i, 0).cwiseProduct(m_F_Transpose_CombinedBlock_col_ordered.block<16, 1>(32, 1).transpose());
        ChunkC3.row(i) = m_SD_precompute_col_ordered.block<1, 16>(i, 0).cwiseProduct(m_F_Transpose_CombinedBlock_col_ordered.block<16, 1>(32, 2).transpose());
        ChunkC4.row(i) = m_SD_precompute_col_ordered.block<1, 16>(i, 16).cwiseProduct(m_F_Transpose_CombinedBlock_col_ordered.block<16, 1>(32, 0).transpose());
        ChunkC5.row(i) = m_SD_precompute_col_ordered.block<1, 16>(i, 16).cwiseProduct(m_F_Transpose_CombinedBlock_col_ordered.block<16, 1>(32, 1).transpose());
        ChunkC6.row(i) = m_SD_precompute_col_ordered.block<1, 16>(i, 16).cwiseProduct(m_F_Transpose_CombinedBlock_col_ordered.block<16, 1>(32, 2).transpose());
        ChunkC7.row(i) = m_SD_precompute_col_ordered.block<1, 16>(i, 32).cwiseProduct(m_F_Transpose_CombinedBlock_col_ordered.block<16, 1>(32, 0).transpose());
        ChunkC8.row(i) = m_SD_precompute_col_ordered.block<1, 16>(i, 32).cwiseProduct(m_F_Transpose_CombinedBlock_col_ordered.block<16, 1>(32, 1).transpose());
        ChunkC9.row(i) = m_SD_precompute_col_ordered.block<1, 16>(i, 32).cwiseProduct(m_F_Transpose_CombinedBlock_col_ordered.block<16, 1>(32, 2).transpose());

        ChunkDvA1.row(i) = m_SD_precompute_col_ordered.block<1, 4>(i, 48).cwiseProduct(m_F_Transpose_CombinedBlock_col_ordered.block<4, 1>(48, 0).transpose());
        ChunkDvA2.row(i) = m_SD_precompute_col_ordered.block<1, 4>(i, 48).cwiseProduct(m_F_Transpose_CombinedBlock_col_ordered.block<4, 1>(48, 1).transpose());
        ChunkDvA3.row(i) = m_SD_precompute_col_ordered.block<1, 4>(i, 48).cwiseProduct(m_F_Transpose_CombinedBlock_col_ordered.block<4, 1>(48, 2).transpose());

        ChunkDvB4.row(i) = m_SD_precompute_col_ordered.block<1, 4>(i, 52).cwiseProduct(m_F_Transpose_CombinedBlock_col_ordered.block<4, 1>(52, 0).transpose());
        ChunkDvB5.row(i) = m_SD_precompute_col_ordered.block<1, 4>(i, 52).cwiseProduct(m_F_Transpose_CombinedBlock_col_ordered.block<4, 1>(52, 1).transpose());
        ChunkDvB6.row(i) = m_SD_precompute_col_ordered.block<1, 4>(i, 52).cwiseProduct(m_F_Transpose_CombinedBlock_col_ordered.block<4, 1>(52, 2).transpose());

        ChunkDvC7.row(i) = m_SD_precompute_col_ordered.block<1, 4>(i, 56).cwiseProduct(m_F_Transpose_CombinedBlock_col_ordered.block<4, 1>(56, 0).transpose());
        ChunkDvC8.row(i) = m_SD_precompute_col_ordered.block<1, 4>(i, 56).cwiseProduct(m_F_Transpose_CombinedBlock_col_ordered.block<4, 1>(56, 1).transpose());
        ChunkDvC9.row(i) = m_SD_precompute_col_ordered.block<1, 4>(i, 56).cwiseProduct(m_F_Transpose_CombinedBlock_col_ordered.block<4, 1>(56, 2).transpose());
    }


    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor, 9, 16> ChunkSA1;
    ChunkSA1.resize(9, 16);
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor, 9, 16> ChunkSA2;
    ChunkSA2.resize(9, 16);
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor, 9, 16> ChunkSA3;
    ChunkSA3.resize(9, 16);
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor, 9, 16> ChunkSA4;
    ChunkSA4.resize(9, 16);
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor, 9, 16> ChunkSA5;
    ChunkSA5.resize(9, 16);
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor, 9, 16> ChunkSA6;
    ChunkSA6.resize(9, 16);
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor, 9, 16> ChunkSA7;
    ChunkSA7.resize(9, 16);
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor, 9, 16> ChunkSA8;
    ChunkSA8.resize(9, 16);
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor, 9, 16> ChunkSA9;
    ChunkSA9.resize(9, 16);

    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor, 9, 16> ChunkSB1;
    ChunkSB1.resize(9, 16);
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor, 9, 16> ChunkSB2;
    ChunkSB2.resize(9, 16);
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor, 9, 16> ChunkSB3;
    ChunkSB3.resize(9, 16);
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor, 9, 16> ChunkSB4;
    ChunkSB4.resize(9, 16);
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor, 9, 16> ChunkSB5;
    ChunkSB5.resize(9, 16);
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor, 9, 16> ChunkSB6;
    ChunkSB6.resize(9, 16);
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor, 9, 16> ChunkSB7;
    ChunkSB7.resize(9, 16);
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor, 9, 16> ChunkSB8;
    ChunkSB8.resize(9, 16);
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor, 9, 16> ChunkSB9;
    ChunkSB9.resize(9, 16);

    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor, 9, 16> ChunkSC1;
    ChunkSC1.resize(9, 16);
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor, 9, 16> ChunkSC2;
    ChunkSC2.resize(9, 16);
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor, 9, 16> ChunkSC3;
    ChunkSC3.resize(9, 16);
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor, 9, 16> ChunkSC4;
    ChunkSC4.resize(9, 16);
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor, 9, 16> ChunkSC5;
    ChunkSC5.resize(9, 16);
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor, 9, 16> ChunkSC6;
    ChunkSC6.resize(9, 16);
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor, 9, 16> ChunkSC7;
    ChunkSC7.resize(9, 16);
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor, 9, 16> ChunkSC8;
    ChunkSC8.resize(9, 16);
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor, 9, 16> ChunkSC9;
    ChunkSC9.resize(9, 16);

    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor, 9, 4> ChunkSDvA1;
    ChunkSDvA1.resize(9, 4);
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor, 9, 4> ChunkSDvA2;
    ChunkSDvA2.resize(9, 4);
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor, 9, 4> ChunkSDvA3;
    ChunkSDvA3.resize(9, 4);

    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor, 9, 4> ChunkSDvB4;
    ChunkSDvB4.resize(9, 4);
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor, 9, 4> ChunkSDvB5;
    ChunkSDvB5.resize(9, 4);
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor, 9, 4> ChunkSDvB6;
    ChunkSDvB6.resize(9, 4);

    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor, 9, 4> ChunkSDvC7;
    ChunkSDvC7.resize(9, 4);
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor, 9, 4> ChunkSDvC8;
    ChunkSDvC8.resize(9, 4);
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor, 9, 4> ChunkSDvC9;
    ChunkSDvC9.resize(9, 4);

    for (auto i = 0; i < 9; i++) {
        ChunkSA1.row(i) = m_SD_precompute_col_ordered.block<1, 16>(i, 0).cwiseProduct(m_GQWeight_det_J_0xi_D0.transpose().cwiseProduct(Kfactor*m_F_Transpose_CombinedBlock_col_ordered.block<16, 1>(0, 0).transpose()));
        ChunkSA2.row(i) = m_SD_precompute_col_ordered.block<1, 16>(i, 0).cwiseProduct(m_GQWeight_det_J_0xi_D0.transpose().cwiseProduct(Kfactor*m_F_Transpose_CombinedBlock_col_ordered.block<16, 1>(0, 1).transpose()));
        ChunkSA3.row(i) = m_SD_precompute_col_ordered.block<1, 16>(i, 0).cwiseProduct(m_GQWeight_det_J_0xi_D0.transpose().cwiseProduct(Kfactor*m_F_Transpose_CombinedBlock_col_ordered.block<16, 1>(0, 2).transpose()));
        ChunkSA4.row(i) = m_SD_precompute_col_ordered.block<1, 16>(i, 16).cwiseProduct(m_GQWeight_det_J_0xi_D0.transpose().cwiseProduct(Kfactor*m_F_Transpose_CombinedBlock_col_ordered.block<16, 1>(0, 0).transpose()));
        ChunkSA5.row(i) = m_SD_precompute_col_ordered.block<1, 16>(i, 16).cwiseProduct(m_GQWeight_det_J_0xi_D0.transpose().cwiseProduct(Kfactor*m_F_Transpose_CombinedBlock_col_ordered.block<16, 1>(0, 1).transpose()));
        ChunkSA6.row(i) = m_SD_precompute_col_ordered.block<1, 16>(i, 16).cwiseProduct(m_GQWeight_det_J_0xi_D0.transpose().cwiseProduct(Kfactor*m_F_Transpose_CombinedBlock_col_ordered.block<16, 1>(0, 2).transpose()));
        ChunkSA7.row(i) = m_SD_precompute_col_ordered.block<1, 16>(i, 32).cwiseProduct(m_GQWeight_det_J_0xi_D0.transpose().cwiseProduct(Kfactor*m_F_Transpose_CombinedBlock_col_ordered.block<16, 1>(0, 0).transpose()));
        ChunkSA8.row(i) = m_SD_precompute_col_ordered.block<1, 16>(i, 32).cwiseProduct(m_GQWeight_det_J_0xi_D0.transpose().cwiseProduct(Kfactor*m_F_Transpose_CombinedBlock_col_ordered.block<16, 1>(0, 1).transpose()));
        ChunkSA9.row(i) = m_SD_precompute_col_ordered.block<1, 16>(i, 32).cwiseProduct(m_GQWeight_det_J_0xi_D0.transpose().cwiseProduct(Kfactor*m_F_Transpose_CombinedBlock_col_ordered.block<16, 1>(0, 2).transpose()));

        ChunkSB1.row(i) = m_SD_precompute_col_ordered.block<1, 16>(i, 0).cwiseProduct(m_GQWeight_det_J_0xi_D0.transpose().cwiseProduct(Kfactor*m_F_Transpose_CombinedBlock_col_ordered.block<16, 1>(16, 0).transpose()));
        ChunkSB2.row(i) = m_SD_precompute_col_ordered.block<1, 16>(i, 0).cwiseProduct(m_GQWeight_det_J_0xi_D0.transpose().cwiseProduct(Kfactor*m_F_Transpose_CombinedBlock_col_ordered.block<16, 1>(16, 1).transpose()));
        ChunkSB3.row(i) = m_SD_precompute_col_ordered.block<1, 16>(i, 0).cwiseProduct(m_GQWeight_det_J_0xi_D0.transpose().cwiseProduct(Kfactor*m_F_Transpose_CombinedBlock_col_ordered.block<16, 1>(16, 2).transpose()));
        ChunkSB4.row(i) = m_SD_precompute_col_ordered.block<1, 16>(i, 16).cwiseProduct(m_GQWeight_det_J_0xi_D0.transpose().cwiseProduct(Kfactor*m_F_Transpose_CombinedBlock_col_ordered.block<16, 1>(16, 0).transpose()));
        ChunkSB5.row(i) = m_SD_precompute_col_ordered.block<1, 16>(i, 16).cwiseProduct(m_GQWeight_det_J_0xi_D0.transpose().cwiseProduct(Kfactor*m_F_Transpose_CombinedBlock_col_ordered.block<16, 1>(16, 1).transpose()));
        ChunkSB6.row(i) = m_SD_precompute_col_ordered.block<1, 16>(i, 16).cwiseProduct(m_GQWeight_det_J_0xi_D0.transpose().cwiseProduct(Kfactor*m_F_Transpose_CombinedBlock_col_ordered.block<16, 1>(16, 2).transpose()));
        ChunkSB7.row(i) = m_SD_precompute_col_ordered.block<1, 16>(i, 32).cwiseProduct(m_GQWeight_det_J_0xi_D0.transpose().cwiseProduct(Kfactor*m_F_Transpose_CombinedBlock_col_ordered.block<16, 1>(16, 0).transpose()));
        ChunkSB8.row(i) = m_SD_precompute_col_ordered.block<1, 16>(i, 32).cwiseProduct(m_GQWeight_det_J_0xi_D0.transpose().cwiseProduct(Kfactor*m_F_Transpose_CombinedBlock_col_ordered.block<16, 1>(16, 1).transpose()));
        ChunkSB9.row(i) = m_SD_precompute_col_ordered.block<1, 16>(i, 32).cwiseProduct(m_GQWeight_det_J_0xi_D0.transpose().cwiseProduct(Kfactor*m_F_Transpose_CombinedBlock_col_ordered.block<16, 1>(16, 2).transpose()));

        ChunkSC1.row(i) = m_SD_precompute_col_ordered.block<1, 16>(i, 0).cwiseProduct(m_GQWeight_det_J_0xi_D0.transpose().cwiseProduct(Kfactor*m_F_Transpose_CombinedBlock_col_ordered.block<16, 1>(32, 0).transpose()));
        ChunkSC2.row(i) = m_SD_precompute_col_ordered.block<1, 16>(i, 0).cwiseProduct(m_GQWeight_det_J_0xi_D0.transpose().cwiseProduct(Kfactor*m_F_Transpose_CombinedBlock_col_ordered.block<16, 1>(32, 1).transpose()));
        ChunkSC3.row(i) = m_SD_precompute_col_ordered.block<1, 16>(i, 0).cwiseProduct(m_GQWeight_det_J_0xi_D0.transpose().cwiseProduct(Kfactor*m_F_Transpose_CombinedBlock_col_ordered.block<16, 1>(32, 2).transpose()));
        ChunkSC4.row(i) = m_SD_precompute_col_ordered.block<1, 16>(i, 16).cwiseProduct(m_GQWeight_det_J_0xi_D0.transpose().cwiseProduct(Kfactor*m_F_Transpose_CombinedBlock_col_ordered.block<16, 1>(32, 0).transpose()));
        ChunkSC5.row(i) = m_SD_precompute_col_ordered.block<1, 16>(i, 16).cwiseProduct(m_GQWeight_det_J_0xi_D0.transpose().cwiseProduct(Kfactor*m_F_Transpose_CombinedBlock_col_ordered.block<16, 1>(32, 1).transpose()));
        ChunkSC6.row(i) = m_SD_precompute_col_ordered.block<1, 16>(i, 16).cwiseProduct(m_GQWeight_det_J_0xi_D0.transpose().cwiseProduct(Kfactor*m_F_Transpose_CombinedBlock_col_ordered.block<16, 1>(32, 2).transpose()));
        ChunkSC7.row(i) = m_SD_precompute_col_ordered.block<1, 16>(i, 32).cwiseProduct(m_GQWeight_det_J_0xi_D0.transpose().cwiseProduct(Kfactor*m_F_Transpose_CombinedBlock_col_ordered.block<16, 1>(32, 0).transpose()));
        ChunkSC8.row(i) = m_SD_precompute_col_ordered.block<1, 16>(i, 32).cwiseProduct(m_GQWeight_det_J_0xi_D0.transpose().cwiseProduct(Kfactor*m_F_Transpose_CombinedBlock_col_ordered.block<16, 1>(32, 1).transpose()));
        ChunkSC9.row(i) = m_SD_precompute_col_ordered.block<1, 16>(i, 32).cwiseProduct(m_GQWeight_det_J_0xi_D0.transpose().cwiseProduct(Kfactor*m_F_Transpose_CombinedBlock_col_ordered.block<16, 1>(32, 2).transpose()));

        ChunkSDvA1.row(i) = m_SD_precompute_col_ordered.block<1, 4>(i, 48).cwiseProduct(m_GQWeight_det_J_0xi_Dv.transpose().cwiseProduct(Kfactor*m_F_Transpose_CombinedBlock_col_ordered.block<4, 1>(48, 0).transpose()));
        ChunkSDvA2.row(i) = m_SD_precompute_col_ordered.block<1, 4>(i, 48).cwiseProduct(m_GQWeight_det_J_0xi_Dv.transpose().cwiseProduct(Kfactor*m_F_Transpose_CombinedBlock_col_ordered.block<4, 1>(48, 1).transpose()));
        ChunkSDvA3.row(i) = m_SD_precompute_col_ordered.block<1, 4>(i, 48).cwiseProduct(m_GQWeight_det_J_0xi_Dv.transpose().cwiseProduct(Kfactor*m_F_Transpose_CombinedBlock_col_ordered.block<4, 1>(48, 2).transpose()));

        ChunkSDvB4.row(i) = m_SD_precompute_col_ordered.block<1, 4>(i, 52).cwiseProduct(m_GQWeight_det_J_0xi_Dv.transpose().cwiseProduct(Kfactor*m_F_Transpose_CombinedBlock_col_ordered.block<4, 1>(52, 0).transpose()));
        ChunkSDvB5.row(i) = m_SD_precompute_col_ordered.block<1, 4>(i, 52).cwiseProduct(m_GQWeight_det_J_0xi_Dv.transpose().cwiseProduct(Kfactor*m_F_Transpose_CombinedBlock_col_ordered.block<4, 1>(52, 1).transpose()));
        ChunkSDvB6.row(i) = m_SD_precompute_col_ordered.block<1, 4>(i, 52).cwiseProduct(m_GQWeight_det_J_0xi_Dv.transpose().cwiseProduct(Kfactor*m_F_Transpose_CombinedBlock_col_ordered.block<4, 1>(52, 2).transpose()));

        ChunkSDvC7.row(i) = m_SD_precompute_col_ordered.block<1, 4>(i, 56).cwiseProduct(m_GQWeight_det_J_0xi_Dv.transpose().cwiseProduct(Kfactor*m_F_Transpose_CombinedBlock_col_ordered.block<4, 1>(56, 0).transpose()));
        ChunkSDvC8.row(i) = m_SD_precompute_col_ordered.block<1, 4>(i, 56).cwiseProduct(m_GQWeight_det_J_0xi_Dv.transpose().cwiseProduct(Kfactor*m_F_Transpose_CombinedBlock_col_ordered.block<4, 1>(56, 1).transpose()));
        ChunkSDvC9.row(i) = m_SD_precompute_col_ordered.block<1, 4>(i, 56).cwiseProduct(m_GQWeight_det_J_0xi_Dv.transpose().cwiseProduct(Kfactor*m_F_Transpose_CombinedBlock_col_ordered.block<4, 1>(56, 2).transpose()));
    }


    //ChMatrixNMc<double, 108, 9> partial_epsilon_partial_e_Block11;
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor, 108, 9> partial_epsilon_partial_e_Block11;
    partial_epsilon_partial_e_Block11.resize(108, 9);
    partial_epsilon_partial_e_Block11.block<16, 9>(0, 0) = ChunkA1.transpose();
    partial_epsilon_partial_e_Block11.block<16, 9>(16, 0) = ChunkB4.transpose();
    partial_epsilon_partial_e_Block11.block<16, 9>(32, 0) = ChunkC7.transpose();
    partial_epsilon_partial_e_Block11.block<16, 9>(48, 0) = ChunkB7.transpose() + ChunkC4.transpose();
    partial_epsilon_partial_e_Block11.block<16, 9>(64, 0) = ChunkA7.transpose() + ChunkC1.transpose();
    partial_epsilon_partial_e_Block11.block<16, 9>(80, 0) = ChunkA4.transpose() + ChunkB1.transpose();
    partial_epsilon_partial_e_Block11.block<4, 9>(96, 0) = ChunkDvA1.transpose();
    partial_epsilon_partial_e_Block11.block<4, 9>(100, 0) = ChunkDvB4.transpose();
    partial_epsilon_partial_e_Block11.block<4, 9>(104, 0) = ChunkDvC7.transpose();

    //ChMatrixNMc<double, 108, 9> partial_epsilon_partial_e_Block22;
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor, 108, 9> partial_epsilon_partial_e_Block22;
    partial_epsilon_partial_e_Block22.resize(108, 9);
    partial_epsilon_partial_e_Block22.block<16, 9>(0, 0) = ChunkA2.transpose();
    partial_epsilon_partial_e_Block22.block<16, 9>(16, 0) = ChunkB5.transpose();
    partial_epsilon_partial_e_Block22.block<16, 9>(32, 0) = ChunkC8.transpose();
    partial_epsilon_partial_e_Block22.block<16, 9>(48, 0) = ChunkB8.transpose() + ChunkC5.transpose();
    partial_epsilon_partial_e_Block22.block<16, 9>(64, 0) = ChunkA8.transpose() + ChunkC2.transpose();
    partial_epsilon_partial_e_Block22.block<16, 9>(80, 0) = ChunkA5.transpose() + ChunkB2.transpose();
    partial_epsilon_partial_e_Block22.block<4, 9>(96, 0) = ChunkDvA2.transpose();
    partial_epsilon_partial_e_Block22.block<4, 9>(100, 0) = ChunkDvB5.transpose();
    partial_epsilon_partial_e_Block22.block<4, 9>(104, 0) = ChunkDvC8.transpose();

    //ChMatrixNMc<double, 108, 9> partial_epsilon_partial_e_Block33;
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor, 108, 9> partial_epsilon_partial_e_Block33;
    partial_epsilon_partial_e_Block33.resize(108, 9);
    partial_epsilon_partial_e_Block33.block<16, 9>(0, 0) = ChunkA3.transpose();
    partial_epsilon_partial_e_Block33.block<16, 9>(16, 0) = ChunkB6.transpose();
    partial_epsilon_partial_e_Block33.block<16, 9>(32, 0) = ChunkC9.transpose();
    partial_epsilon_partial_e_Block33.block<16, 9>(48, 0) = ChunkB9.transpose() + ChunkC6.transpose();
    partial_epsilon_partial_e_Block33.block<16, 9>(64, 0) = ChunkA9.transpose() + ChunkC3.transpose();
    partial_epsilon_partial_e_Block33.block<16, 9>(80, 0) = ChunkA6.transpose() + ChunkB3.transpose();
    partial_epsilon_partial_e_Block33.block<4, 9>(96, 0) = ChunkDvA3.transpose();
    partial_epsilon_partial_e_Block33.block<4, 9>(100, 0) = ChunkDvB6.transpose();
    partial_epsilon_partial_e_Block33.block<4, 9>(104, 0) = ChunkDvC9.transpose();


    ChVectorN<double, 6> D0 = GetMaterial()->Get_D0();
    ChMatrix33<double> Dv = GetMaterial()->Get_Dv();

    //ChMatrixNMc<double, 108, 9> Scaled_Combined_partial_epsilon_partial_e_Block11;
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor, 108, 9> Scaled_Combined_partial_epsilon_partial_e_Block11;
    Scaled_Combined_partial_epsilon_partial_e_Block11.resize(108, 9);
    Scaled_Combined_partial_epsilon_partial_e_Block11.block<16, 9>(0, 0) = D0(0)*ChunkSA1.transpose();
    Scaled_Combined_partial_epsilon_partial_e_Block11.block<16, 9>(16, 0) = D0(1)*ChunkSB4.transpose();
    Scaled_Combined_partial_epsilon_partial_e_Block11.block<16, 9>(32, 0) = D0(2)*ChunkSC7.transpose();
    Scaled_Combined_partial_epsilon_partial_e_Block11.block<16, 9>(48, 0) = D0(3)*(ChunkSB7.transpose() + ChunkSC4.transpose());
    Scaled_Combined_partial_epsilon_partial_e_Block11.block<16, 9>(64, 0) = D0(4)*(ChunkSA7.transpose() + ChunkSC1.transpose());
    Scaled_Combined_partial_epsilon_partial_e_Block11.block<16, 9>(80, 0) = D0(5)*(ChunkSA4.transpose() + ChunkSB1.transpose());
    Scaled_Combined_partial_epsilon_partial_e_Block11.block<4, 9>(96, 0) = Dv(0, 0)*ChunkSDvA1.transpose() + Dv(0, 1)*ChunkSDvB4.transpose() + Dv(0, 2)*ChunkSDvC7.transpose();
    Scaled_Combined_partial_epsilon_partial_e_Block11.block<4, 9>(100, 0) = Dv(1, 0)*ChunkSDvA1.transpose() + Dv(1, 1)*ChunkSDvB4.transpose() + Dv(1, 2)*ChunkSDvC7.transpose();
    Scaled_Combined_partial_epsilon_partial_e_Block11.block<4, 9>(104, 0) = Dv(2, 0)*ChunkSDvA1.transpose() + Dv(2, 1)*ChunkSDvB4.transpose() + Dv(2, 2)*ChunkSDvC7.transpose();

    //ChMatrixNMc<double, 108, 9> Scaled_Combined_partial_epsilon_partial_e_Block22;
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor, 108, 9> Scaled_Combined_partial_epsilon_partial_e_Block22;
    Scaled_Combined_partial_epsilon_partial_e_Block22.resize(108, 9);
    Scaled_Combined_partial_epsilon_partial_e_Block22.block<16, 9>(0, 0) = D0(0)*ChunkSA2.transpose();
    Scaled_Combined_partial_epsilon_partial_e_Block22.block<16, 9>(16, 0) = D0(1)*ChunkSB5.transpose();
    Scaled_Combined_partial_epsilon_partial_e_Block22.block<16, 9>(32, 0) = D0(2)*ChunkSC8.transpose();
    Scaled_Combined_partial_epsilon_partial_e_Block22.block<16, 9>(48, 0) = D0(3)*(ChunkSB8.transpose() + ChunkSC5.transpose());
    Scaled_Combined_partial_epsilon_partial_e_Block22.block<16, 9>(64, 0) = D0(4)*(ChunkSA8.transpose() + ChunkSC2.transpose());
    Scaled_Combined_partial_epsilon_partial_e_Block22.block<16, 9>(80, 0) = D0(5)*(ChunkSA5.transpose() + ChunkSB2.transpose());
    Scaled_Combined_partial_epsilon_partial_e_Block22.block<4, 9>(96, 0) = Dv(0, 0)*ChunkSDvA2.transpose() + Dv(0, 1)*ChunkSDvB5.transpose() + Dv(0, 2)*ChunkSDvC8.transpose();
    Scaled_Combined_partial_epsilon_partial_e_Block22.block<4, 9>(100, 0) = Dv(1, 0)*ChunkSDvA2.transpose() + Dv(1, 1)*ChunkSDvB5.transpose() + Dv(1, 2)*ChunkSDvC8.transpose();
    Scaled_Combined_partial_epsilon_partial_e_Block22.block<4, 9>(104, 0) = Dv(2, 0)*ChunkSDvA2.transpose() + Dv(2, 1)*ChunkSDvB5.transpose() + Dv(2, 2)*ChunkSDvC8.transpose();

    //ChMatrixNMc<double, 108, 9> Scaled_Combined_partial_epsilon_partial_e_Block33;
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor, 108, 9> Scaled_Combined_partial_epsilon_partial_e_Block33;
    Scaled_Combined_partial_epsilon_partial_e_Block33.resize(108, 9);
    Scaled_Combined_partial_epsilon_partial_e_Block33.block<16, 9>(0, 0) = D0(0)*ChunkSA3.transpose();
    Scaled_Combined_partial_epsilon_partial_e_Block33.block<16, 9>(16, 0) = D0(1)*ChunkSB6.transpose();
    Scaled_Combined_partial_epsilon_partial_e_Block33.block<16, 9>(32, 0) = D0(2)*ChunkSC9.transpose();
    Scaled_Combined_partial_epsilon_partial_e_Block33.block<16, 9>(48, 0) = D0(3)*ChunkSB9.transpose() + D0(3)*ChunkSC6.transpose();
    Scaled_Combined_partial_epsilon_partial_e_Block33.block<16, 9>(64, 0) = D0(4)*ChunkSA9.transpose() + D0(4)*ChunkSC3.transpose();
    Scaled_Combined_partial_epsilon_partial_e_Block33.block<16, 9>(80, 0) = D0(5)*ChunkSA6.transpose() + D0(5)*ChunkSB3.transpose();
    Scaled_Combined_partial_epsilon_partial_e_Block33.block<4, 9>(96, 0) = Dv(0, 0)*ChunkSDvA3.transpose() + Dv(0, 1)*ChunkSDvB6.transpose() + Dv(0, 2)*ChunkSDvC9.transpose();
    Scaled_Combined_partial_epsilon_partial_e_Block33.block<4, 9>(100, 0) = Dv(1, 0)*ChunkSDvA3.transpose() + Dv(1, 1)*ChunkSDvB6.transpose() + Dv(1, 2)*ChunkSDvC9.transpose();
    Scaled_Combined_partial_epsilon_partial_e_Block33.block<4, 9>(104, 0) = Dv(2, 0)*ChunkSDvA3.transpose() + Dv(2, 1)*ChunkSDvB6.transpose() + Dv(2, 2)*ChunkSDvC9.transpose();


    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor, 9, 9> Block11;
    Block11.resize(9, 9);
    Block11 = partial_epsilon_partial_e_Block11.transpose()*Scaled_Combined_partial_epsilon_partial_e_Block11;
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor, 9, 9> Block12;
    Block12.resize(9, 9);
    Block12 = partial_epsilon_partial_e_Block11.transpose()*Scaled_Combined_partial_epsilon_partial_e_Block22;
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor, 9, 9> Block13;
    Block13.resize(9, 9);
    Block13 = partial_epsilon_partial_e_Block11.transpose()*Scaled_Combined_partial_epsilon_partial_e_Block33;

    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor, 9, 9> Block21;
    Block21.resize(9, 9);
    Block21 = partial_epsilon_partial_e_Block22.transpose()*Scaled_Combined_partial_epsilon_partial_e_Block11;
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor, 9, 9> Block22;
    Block22.resize(9, 9);
    Block22 = partial_epsilon_partial_e_Block22.transpose()*Scaled_Combined_partial_epsilon_partial_e_Block22;
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor, 9, 9> Block23;
    Block23.resize(9, 9);
    Block23 = partial_epsilon_partial_e_Block22.transpose()*Scaled_Combined_partial_epsilon_partial_e_Block33;

    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor, 9, 9> Block31;
    Block31.resize(9, 9);
    Block31 = partial_epsilon_partial_e_Block33.transpose()*Scaled_Combined_partial_epsilon_partial_e_Block11;
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor, 9, 9> Block32;
    Block32.resize(9, 9);
    Block32 = partial_epsilon_partial_e_Block33.transpose()*Scaled_Combined_partial_epsilon_partial_e_Block22;
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor, 9, 9> Block33;
    Block33.resize(9, 9);
    Block33 = partial_epsilon_partial_e_Block33.transpose()*Scaled_Combined_partial_epsilon_partial_e_Block33;


    for (unsigned int i = 0; i < 9; i++) {
        for (unsigned int j = 0; j < 9; j++) {
            H(3 * i, 3 * j) = -Block11(i, j) + Jacobian_CompactPart(i, j);
            H(3 * i + 1, 3 * j) = -Block21(i, j);
            H(3 * i + 2, 3 * j) = -Block31(i, j);

            H(3 * i, 3 * j + 1) = -Block12(i, j);
            H(3 * i + 1, 3 * j + 1) = -Block22(i, j) + Jacobian_CompactPart(i, j);
            H(3 * i + 2, 3 * j + 1) = -Block32(i, j);

            H(3 * i, 3 * j + 2) = -Block13(i, j);
            H(3 * i + 1, 3 * j + 2) = -Block23(i, j);
            H(3 * i + 2, 3 * j + 2) = -Block33(i, j) + Jacobian_CompactPart(i, j);
        }
    }
}



// Return the mass matrix.
void ChElementBeamANCF_MT35::ComputeMmatrixGlobal(ChMatrixRef M) {
    M.setZero();

    //Inflate the Mass Matrix since it is stored in compact form.
    //In MATLAB notation:	 
    //M(1:3:end,1:3:end) = m_MassMatrix;
    //M(2:3:end,2:3:end) = m_MassMatrix;
    //M(3:3:end,3:3:end) = m_MassMatrix;
    for (unsigned int i = 0; i < 9; i++) {
        for (unsigned int j = 0; j < 9; j++) {
            M(3 * i, 3 * j) = m_MassMatrix(i, j);
            M(3 * i + 1, 3 * j + 1) = m_MassMatrix(i, j);
            M(3 * i + 2, 3 * j + 2) = m_MassMatrix(i, j);
        }
    }
}

// -----------------------------------------------------------------------------
// Mass Matrix & Generalized Force Due to Gravity Calculation
// -----------------------------------------------------------------------------
void ChElementBeamANCF_MT35::ComputeMassMatrixAndGravityForce(const ChVector<>& g_acc) {
    // For this element, 5 GQ Points are needed in the xi direction
    //  and 2 GQ Points are needed in the eta & zeta directions
    //  for exact integration of the element's mass matrix, even if 
    //  the reference configuration is not straight
    // Since the major pieces of the generalized force due to gravity
    //  can also be used to calculate the mass matrix, these calculations
    //  are performed at the same time.

    ChQuadratureTables* GQTable = GetStaticGQTables();
    unsigned int GQ_idx_xi = 4;       // 5 Point Gauss-Quadrature;
    unsigned int GQ_idx_eta_zeta = 1; // 2 Point Gauss-Quadrature;

    double rho = GetMaterial()->Get_rho(); //Density of the material for the element

    //Set these to zeros since they will be incremented as the vector/matrix is calculated
    m_MassMatrix.setZero();
    m_GravForce.setZero();

    for (unsigned int it_xi = 0; it_xi < GQTable->Lroots[GQ_idx_xi].size(); it_xi++) {
        for (unsigned int it_eta = 0; it_eta < GQTable->Lroots[GQ_idx_eta_zeta].size(); it_eta++) {
            for (unsigned int it_zeta = 0; it_zeta < GQTable->Lroots[GQ_idx_eta_zeta].size(); it_zeta++) {
                double GQ_weight = GQTable->Weight[GQ_idx_xi][it_xi] * GQTable->Weight[GQ_idx_eta_zeta][it_eta] * GQTable->Weight[GQ_idx_eta_zeta][it_zeta];
                double xi = GQTable->Lroots[GQ_idx_xi][it_xi];
                double eta = GQTable->Lroots[GQ_idx_eta_zeta][it_eta];
                double zeta = GQTable->Lroots[GQ_idx_eta_zeta][it_zeta];
                double det_J_0xi = Calc_det_J_0xi(xi, eta, zeta); // determinate of the element Jacobian (volume ratio)
                ChMatrixNM<double, 3, 27> Sxi; //3x27 Normalized Shape Function Matrix
                Calc_Sxi(Sxi, xi, eta, zeta);
                ChVectorN<double, 9> Sxi_compact; //9x1 Vector of the Unique Normalized Shape Functions
                Calc_Sxi_compact(Sxi_compact, xi, eta, zeta);

                m_GravForce += (GQ_weight*rho*det_J_0xi)*Sxi.transpose()*g_acc.eigen();
                m_MassMatrix += (GQ_weight*rho*det_J_0xi)*Sxi_compact*Sxi_compact.transpose();
            }
        }
    }

}

//Precalculate constant matrices and scalars for the internal force calculations
void ChElementBeamANCF_MT35::PrecomputeInternalForceMatricesWeights() {
    ChQuadratureTables* GQTable = GetStaticGQTables();
    unsigned int GQ_idx_xi = 3;       // 4 Point Gauss-Quadrature;
    unsigned int GQ_idx_eta_zeta = 1; // 2 Point Gauss-Quadrature;

    //Precalculate the matrices of normalized shape function derivatives corrected for a potentially non-straight reference configuration & GQ Weights times the determiate of the element Jacobian for later
    //Calculating the portion of the Selective Reduced Integration that does account for the Poisson effect
    for (unsigned int it_xi = 0; it_xi < GQTable->Lroots[GQ_idx_xi].size(); it_xi++) {
        for (unsigned int it_eta = 0; it_eta < GQTable->Lroots[GQ_idx_eta_zeta].size(); it_eta++) {
            for (unsigned int it_zeta = 0; it_zeta < GQTable->Lroots[GQ_idx_eta_zeta].size(); it_zeta++) {
                double GQ_weight = GQTable->Weight[GQ_idx_xi][it_xi] * GQTable->Weight[GQ_idx_eta_zeta][it_eta] * GQTable->Weight[GQ_idx_eta_zeta][it_zeta];
                double xi = GQTable->Lroots[GQ_idx_xi][it_xi];
                double eta = GQTable->Lroots[GQ_idx_eta_zeta][it_eta];
                double zeta = GQTable->Lroots[GQ_idx_eta_zeta][it_zeta];
                auto index = it_zeta + it_eta*GQTable->Lroots[GQ_idx_eta_zeta].size() + it_xi*GQTable->Lroots[GQ_idx_eta_zeta].size()*GQTable->Lroots[GQ_idx_eta_zeta].size();
                ChMatrix33<double> J_0xi;               //Element Jacobian between the reference configuration and normalized configuration
                ChMatrixNMc<double, 9, 3> Sxi_D;         //Matrix of normalized shape function derivatives

                Calc_Sxi_D(Sxi_D, xi, eta, zeta);
                J_0xi.noalias() = m_e0_bar.transpose()*Sxi_D;
   
                ChMatrixNMc<double, 9, 3> SD = Sxi_D * J_0xi.inverse();
                m_SD_precompute_D0.block(0, 3*index, 9, 3) = SD;
				m_SD_precompute_D0_col0_block.block(0, index, 9, 1) = SD.col(0);
                m_SD_precompute_D0_col1_block.block(0, index, 9, 1) = SD.col(1);
                m_SD_precompute_D0_col2_block.block(0, index, 9, 1) = SD.col(2);
                m_GQWeight_det_J_0xi_D0(index) = -J_0xi.determinant()*GQ_weight;
            }
        }
    }

    //Precalculate the matrices of normalized shape function derivatives corrected for a potentially non-straight reference configuration & GQ Weights times the determiate of the element Jacobian for later
    //Calculate the portion of the Selective Reduced Integration that account for the Poisson effect, but only on the beam axis
    for (unsigned int it_xi = 0; it_xi < GQTable->Lroots[GQ_idx_xi].size(); it_xi++) {
        double GQ_weight = GQTable->Weight[GQ_idx_xi][it_xi] * 2 * 2;
        double xi = GQTable->Lroots[GQ_idx_xi][it_xi];
        ChMatrix33<double> J_0xi;               //Element Jacobian between the reference configuration and normalized configuration
        ChMatrixNMc<double, 9, 3> Sxi_D;         //Matrix of normalized shape function derivatives

        Calc_Sxi_D(Sxi_D, xi, 0, 0);
        J_0xi.noalias() = m_e0_bar.transpose()*Sxi_D;

		ChMatrixNMc<double, 9, 3> SD = Sxi_D * J_0xi.inverse();
        m_SD_precompute_Dv.block(0, 3*it_xi, 9, 3) = SD;
        m_SD_precompute_Dv_col0_block.block(0, it_xi, 9, 1) = SD.col(0);
        m_SD_precompute_Dv_col1_block.block(0, it_xi, 9, 1) = SD.col(1);
        m_SD_precompute_Dv_col2_block.block(0, it_xi, 9, 1) = SD.col(2);
        m_GQWeight_det_J_0xi_Dv(it_xi) = -J_0xi.determinant()*GQ_weight;
    }

    m_SD_precompute.block(0, 0, 9, 48) = m_SD_precompute_D0;
    m_SD_precompute.block(0, 48, 9, 12) = m_SD_precompute_Dv;

    m_SD_precompute_col_ordered.block(0, 0, 9, 16) = m_SD_precompute_D0_col0_block;
    m_SD_precompute_col_ordered.block(0, 16, 9, 16) = m_SD_precompute_D0_col1_block;
    m_SD_precompute_col_ordered.block(0, 32, 9, 16) = m_SD_precompute_D0_col2_block;
    m_SD_precompute_col_ordered.block(0, 48, 9, 4) = m_SD_precompute_Dv_col0_block;
    m_SD_precompute_col_ordered.block(0, 52, 9, 4) = m_SD_precompute_Dv_col1_block;
    m_SD_precompute_col_ordered.block(0, 56, 9, 4) = m_SD_precompute_Dv_col2_block;

    m_GQWeight_det_J_0xi.block(0, 0, 16, 1) = m_GQWeight_det_J_0xi_D0;
    m_GQWeight_det_J_0xi.block(16, 0, 4, 1) = m_GQWeight_det_J_0xi_Dv;

}

/// This class computes and adds corresponding masses to ElementGeneric member m_TotalMass
void ChElementBeamANCF_MT35::ComputeNodalMass() {
    m_nodes[0]->m_TotalMass += m_MassMatrix(0, 0) + m_MassMatrix(0, 3) + m_MassMatrix(0, 6);
    m_nodes[1]->m_TotalMass += m_MassMatrix(3, 3) + m_MassMatrix(3, 0) + m_MassMatrix(3, 6);
    m_nodes[2]->m_TotalMass += m_MassMatrix(6, 6) + m_MassMatrix(6, 0) + m_MassMatrix(6, 3);
}

// -----------------------------------------------------------------------------
// Elastic force calculation
// -----------------------------------------------------------------------------

// Set structural damping.
void ChElementBeamANCF_MT35::SetAlphaDamp(double a) {
    m_Alpha = a;
    m_2Alpha = 2 * a;
    if (std::abs(m_Alpha) > 1e-10)
        m_damping_enabled = true;
    else
        m_damping_enabled = false;
}

void ChElementBeamANCF_MT35::ComputeInternalForces(ChVectorDynamic<>& Fi) {   
    //Runs faster if the internal force with or without damping calculations are not combined into the same function using the common calculations with an if statement for the damping in the middle to calculate the different P_transpose_scaled_Block components
    if (m_damping_enabled) {
        ChMatrixNMc<double, 9, 6> ebar_ebardot;
        CalcCombinedCoordMatrix(ebar_ebardot);

        ComputeInternalForcesAtState(Fi, ebar_ebardot);
    }
    else {
        ChMatrixNMc<double, 9, 3> e_bar;
        CalcCoordMatrix(e_bar);

        ComputeInternalForcesAtStateNoDamping(Fi, e_bar);
    }

    if (m_gravity_on) {
        Fi += m_GravForce;
    }
}

void ChElementBeamANCF_MT35::ComputeInternalForcesAtState(ChVectorDynamic<>& Fi, const ChMatrixNMc<double, 9, 6>& ebar_ebardot) {

    // Straight & Normalized Internal Force Integrand is of order : 8 in xi, order : 4 in eta, and order : 4 in zeta.
    // This requires GQ 5 points along the xi direction and 3 points along the eta and zeta directions for "Full Integration"
    // However, very similar results can be obtained with 1 fewer GQ point in  each direction, resulting in roughly 1/3 of the calculations

    //Calculate F is one big block and then split up afterwards to improve efficiency (hopefully)
    m_F_Transpose_CombinedBlockDamping_col_ordered.noalias() = m_SD_precompute_col_ordered.transpose()*ebar_ebardot;

    ChMatrixNMc<double, 16, 1> Fcol0_Transpose_D0_Block_col0 = m_F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(0, 0);
	ChMatrixNMc<double, 16, 1> Fcol0_Transpose_D0_Block_col1 = m_F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(0, 1);
	ChMatrixNMc<double, 16, 1> Fcol0_Transpose_D0_Block_col2 = m_F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(0, 2);
    ChMatrixNMc<double, 16, 1> Fdotcol0_Transpose_D0_Block_col0 = m_F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(0, 3);
	ChMatrixNMc<double, 16, 1> Fdotcol0_Transpose_D0_Block_col1 = m_F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(0, 4);
	ChMatrixNMc<double, 16, 1> Fdotcol0_Transpose_D0_Block_col2 = m_F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(0, 5);
	
    ChMatrixNMc<double, 16, 1> Fcol1_Transpose_D0_Block_col0 = m_F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(16, 0);
	ChMatrixNMc<double, 16, 1> Fcol1_Transpose_D0_Block_col1 = m_F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(16, 1);
	ChMatrixNMc<double, 16, 1> Fcol1_Transpose_D0_Block_col2 = m_F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(16, 2);
    ChMatrixNMc<double, 16, 1> Fdotcol1_Transpose_D0_Block_col0 = m_F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(16, 3);
	ChMatrixNMc<double, 16, 1> Fdotcol1_Transpose_D0_Block_col1 = m_F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(16, 4);
	ChMatrixNMc<double, 16, 1> Fdotcol1_Transpose_D0_Block_col2 = m_F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(16, 5);
	
    ChMatrixNMc<double, 16, 1> Fcol2_Transpose_D0_Block_col0 = m_F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(32, 0);
	ChMatrixNMc<double, 16, 1> Fcol2_Transpose_D0_Block_col1 = m_F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(32, 1);
	ChMatrixNMc<double, 16, 1> Fcol2_Transpose_D0_Block_col2 = m_F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(32, 2);
    ChMatrixNMc<double, 16, 1> Fdotcol2_Transpose_D0_Block_col0 = m_F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(32, 3);
	ChMatrixNMc<double, 16, 1> Fdotcol2_Transpose_D0_Block_col1 = m_F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(32, 4);
	ChMatrixNMc<double, 16, 1> Fdotcol2_Transpose_D0_Block_col2 = m_F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(32, 5);

    ChMatrixNMc<double, 4, 1> Fcol0_Transpose_Dv_Block_col0 = m_F_Transpose_CombinedBlockDamping_col_ordered.block<4, 1>(48, 0);
	ChMatrixNMc<double, 4, 1> Fcol0_Transpose_Dv_Block_col1 = m_F_Transpose_CombinedBlockDamping_col_ordered.block<4, 1>(48, 1);
	ChMatrixNMc<double, 4, 1> Fcol0_Transpose_Dv_Block_col2 = m_F_Transpose_CombinedBlockDamping_col_ordered.block<4, 1>(48, 2);
    ChMatrixNMc<double, 4, 1> Fdotcol0_Transpose_Dv_Block_col0 = m_F_Transpose_CombinedBlockDamping_col_ordered.block<4, 1>(48, 3);
	ChMatrixNMc<double, 4, 1> Fdotcol0_Transpose_Dv_Block_col1 = m_F_Transpose_CombinedBlockDamping_col_ordered.block<4, 1>(48, 4);
	ChMatrixNMc<double, 4, 1> Fdotcol0_Transpose_Dv_Block_col2 = m_F_Transpose_CombinedBlockDamping_col_ordered.block<4, 1>(48, 5);
	
    ChMatrixNMc<double, 4, 1> Fcol1_Transpose_Dv_Block_col0 = m_F_Transpose_CombinedBlockDamping_col_ordered.block<4, 1>(52, 0);
	ChMatrixNMc<double, 4, 1> Fcol1_Transpose_Dv_Block_col1 = m_F_Transpose_CombinedBlockDamping_col_ordered.block<4, 1>(52, 1);
	ChMatrixNMc<double, 4, 1> Fcol1_Transpose_Dv_Block_col2 = m_F_Transpose_CombinedBlockDamping_col_ordered.block<4, 1>(52, 2);
    ChMatrixNMc<double, 4, 1> Fdotcol1_Transpose_Dv_Block_col0 = m_F_Transpose_CombinedBlockDamping_col_ordered.block<4, 1>(52, 3);
	ChMatrixNMc<double, 4, 1> Fdotcol1_Transpose_Dv_Block_col1 = m_F_Transpose_CombinedBlockDamping_col_ordered.block<4, 1>(52, 4);
	ChMatrixNMc<double, 4, 1> Fdotcol1_Transpose_Dv_Block_col2 = m_F_Transpose_CombinedBlockDamping_col_ordered.block<4, 1>(52, 5);
    
	ChMatrixNMc<double, 4, 1> Fcol2_Transpose_Dv_Block_col0 = m_F_Transpose_CombinedBlockDamping_col_ordered.block<4, 1>(56, 0);
    ChMatrixNMc<double, 4, 1> Fcol2_Transpose_Dv_Block_col1 = m_F_Transpose_CombinedBlockDamping_col_ordered.block<4, 1>(56, 1);
	ChMatrixNMc<double, 4, 1> Fcol2_Transpose_Dv_Block_col2 = m_F_Transpose_CombinedBlockDamping_col_ordered.block<4, 1>(56, 2);
    ChMatrixNMc<double, 4, 1> Fdotcol2_Transpose_Dv_Block_col0 = m_F_Transpose_CombinedBlockDamping_col_ordered.block<4, 1>(56, 3);
    ChMatrixNMc<double, 4, 1> Fdotcol2_Transpose_Dv_Block_col1 = m_F_Transpose_CombinedBlockDamping_col_ordered.block<4, 1>(56, 4);
	ChMatrixNMc<double, 4, 1> Fdotcol2_Transpose_Dv_Block_col2 = m_F_Transpose_CombinedBlockDamping_col_ordered.block<4, 1>(56, 5);

    ChVectorN<double, 6> D0 = GetMaterial()->Get_D0();

    m_SPK2_0_D0_Block.noalias() = Fcol0_Transpose_D0_Block_col0.cwiseProduct(Fcol0_Transpose_D0_Block_col0);
    m_SPK2_0_D0_Block += Fcol0_Transpose_D0_Block_col1.cwiseProduct(Fcol0_Transpose_D0_Block_col1);
    m_SPK2_0_D0_Block += Fcol0_Transpose_D0_Block_col2.cwiseProduct(Fcol0_Transpose_D0_Block_col2);
    m_SPK2_0_D0_Block.array() -= 1;
    ChVectorN<double, 16> SPK2_0_D0_BlockDamping = Fcol0_Transpose_D0_Block_col0.cwiseProduct(Fdotcol0_Transpose_D0_Block_col0);
    SPK2_0_D0_BlockDamping += Fcol0_Transpose_D0_Block_col1.cwiseProduct(Fdotcol0_Transpose_D0_Block_col1);
    SPK2_0_D0_BlockDamping += Fcol0_Transpose_D0_Block_col2.cwiseProduct(Fdotcol0_Transpose_D0_Block_col2);
    m_SPK2_0_D0_Block += m_2Alpha * SPK2_0_D0_BlockDamping;
    m_SPK2_0_D0_Block.array() *= m_GQWeight_det_J_0xi_D0.array();
    m_SPK2_0_D0_Block *= (0.5 * D0(0));

    m_SPK2_1_D0_Block.noalias() = Fcol1_Transpose_D0_Block_col0.cwiseProduct(Fcol1_Transpose_D0_Block_col0);
    m_SPK2_1_D0_Block += Fcol1_Transpose_D0_Block_col1.cwiseProduct(Fcol1_Transpose_D0_Block_col1);
    m_SPK2_1_D0_Block += Fcol1_Transpose_D0_Block_col2.cwiseProduct(Fcol1_Transpose_D0_Block_col2);
    m_SPK2_1_D0_Block.array() -= 1;
    ChVectorN<double, 16> SPK2_1_D0_BlockDamping = Fcol1_Transpose_D0_Block_col0.cwiseProduct(Fdotcol1_Transpose_D0_Block_col0);
    SPK2_1_D0_BlockDamping += Fcol1_Transpose_D0_Block_col1.cwiseProduct(Fdotcol1_Transpose_D0_Block_col1);
    SPK2_1_D0_BlockDamping += Fcol1_Transpose_D0_Block_col2.cwiseProduct(Fdotcol1_Transpose_D0_Block_col2);
    m_SPK2_1_D0_Block += m_2Alpha * SPK2_1_D0_BlockDamping;
    m_SPK2_1_D0_Block.array() *= m_GQWeight_det_J_0xi_D0.array();
    m_SPK2_1_D0_Block *= (0.5 * D0(1));

    m_SPK2_2_D0_Block.noalias() = Fcol2_Transpose_D0_Block_col0.cwiseProduct(Fcol2_Transpose_D0_Block_col0);
    m_SPK2_2_D0_Block += Fcol2_Transpose_D0_Block_col1.cwiseProduct(Fcol2_Transpose_D0_Block_col1);
    m_SPK2_2_D0_Block += Fcol2_Transpose_D0_Block_col2.cwiseProduct(Fcol2_Transpose_D0_Block_col2);
    m_SPK2_2_D0_Block.array() -= 1;
    ChVectorN<double, 16> SPK2_2_D0_BlockDamping = Fcol2_Transpose_D0_Block_col0.cwiseProduct(Fdotcol2_Transpose_D0_Block_col0);
    SPK2_2_D0_BlockDamping += Fcol2_Transpose_D0_Block_col1.cwiseProduct(Fdotcol2_Transpose_D0_Block_col1);
    SPK2_2_D0_BlockDamping += Fcol2_Transpose_D0_Block_col2.cwiseProduct(Fdotcol2_Transpose_D0_Block_col2);
    m_SPK2_2_D0_Block += m_2Alpha * SPK2_2_D0_BlockDamping;
    m_SPK2_2_D0_Block.array() *= m_GQWeight_det_J_0xi_D0.array();
    m_SPK2_2_D0_Block *= (0.5 * D0(2));

    m_SPK2_3_D0_Block.noalias() = Fcol1_Transpose_D0_Block_col0.cwiseProduct(Fcol2_Transpose_D0_Block_col0);
    m_SPK2_3_D0_Block += Fcol1_Transpose_D0_Block_col1.cwiseProduct(Fcol2_Transpose_D0_Block_col1);
    m_SPK2_3_D0_Block += Fcol1_Transpose_D0_Block_col2.cwiseProduct(Fcol2_Transpose_D0_Block_col2);
    ChVectorN<double, 16> SPK2_3_D0_BlockDamping = Fcol2_Transpose_D0_Block_col0.cwiseProduct(Fdotcol1_Transpose_D0_Block_col0);
    SPK2_3_D0_BlockDamping += Fcol2_Transpose_D0_Block_col1.cwiseProduct(Fdotcol1_Transpose_D0_Block_col1);
    SPK2_3_D0_BlockDamping += Fcol2_Transpose_D0_Block_col2.cwiseProduct(Fdotcol1_Transpose_D0_Block_col2);
    SPK2_3_D0_BlockDamping += Fcol1_Transpose_D0_Block_col0.cwiseProduct(Fdotcol2_Transpose_D0_Block_col0);
    SPK2_3_D0_BlockDamping += Fcol1_Transpose_D0_Block_col1.cwiseProduct(Fdotcol2_Transpose_D0_Block_col1);
    SPK2_3_D0_BlockDamping += Fcol1_Transpose_D0_Block_col2.cwiseProduct(Fdotcol2_Transpose_D0_Block_col2);
    m_SPK2_3_D0_Block += m_Alpha * SPK2_3_D0_BlockDamping;
    m_SPK2_3_D0_Block.array() *= m_GQWeight_det_J_0xi_D0.array();
    m_SPK2_3_D0_Block *= D0(3);

    m_SPK2_4_D0_Block.noalias() = Fcol0_Transpose_D0_Block_col0.cwiseProduct(Fcol2_Transpose_D0_Block_col0);
    m_SPK2_4_D0_Block += Fcol0_Transpose_D0_Block_col1.cwiseProduct(Fcol2_Transpose_D0_Block_col1);
    m_SPK2_4_D0_Block += Fcol0_Transpose_D0_Block_col2.cwiseProduct(Fcol2_Transpose_D0_Block_col2);
    ChVectorN<double, 16> SPK2_4_D0_BlockDamping = Fcol2_Transpose_D0_Block_col0.cwiseProduct(Fdotcol0_Transpose_D0_Block_col0);
    SPK2_4_D0_BlockDamping += Fcol2_Transpose_D0_Block_col1.cwiseProduct(Fdotcol0_Transpose_D0_Block_col1);
    SPK2_4_D0_BlockDamping += Fcol2_Transpose_D0_Block_col2.cwiseProduct(Fdotcol0_Transpose_D0_Block_col2);
    SPK2_4_D0_BlockDamping += Fcol0_Transpose_D0_Block_col0.cwiseProduct(Fdotcol2_Transpose_D0_Block_col0);
    SPK2_4_D0_BlockDamping += Fcol0_Transpose_D0_Block_col1.cwiseProduct(Fdotcol2_Transpose_D0_Block_col1);
    SPK2_4_D0_BlockDamping += Fcol0_Transpose_D0_Block_col2.cwiseProduct(Fdotcol2_Transpose_D0_Block_col2);
    m_SPK2_4_D0_Block += m_Alpha * SPK2_4_D0_BlockDamping;
    m_SPK2_4_D0_Block.array() *= m_GQWeight_det_J_0xi_D0.array();
    m_SPK2_4_D0_Block *= D0(4);

    m_SPK2_5_D0_Block.noalias() = Fcol0_Transpose_D0_Block_col0.cwiseProduct(Fcol1_Transpose_D0_Block_col0);
    m_SPK2_5_D0_Block += Fcol0_Transpose_D0_Block_col1.cwiseProduct(Fcol1_Transpose_D0_Block_col1);
    m_SPK2_5_D0_Block += Fcol0_Transpose_D0_Block_col2.cwiseProduct(Fcol1_Transpose_D0_Block_col2);
    ChVectorN<double, 16> SPK2_5_D0_BlockDamping = Fcol1_Transpose_D0_Block_col0.cwiseProduct(Fdotcol0_Transpose_D0_Block_col0);
    SPK2_5_D0_BlockDamping += Fcol1_Transpose_D0_Block_col1.cwiseProduct(Fdotcol0_Transpose_D0_Block_col1);
    SPK2_5_D0_BlockDamping += Fcol1_Transpose_D0_Block_col2.cwiseProduct(Fdotcol0_Transpose_D0_Block_col2);
    SPK2_5_D0_BlockDamping += Fcol0_Transpose_D0_Block_col0.cwiseProduct(Fdotcol1_Transpose_D0_Block_col0);
    SPK2_5_D0_BlockDamping += Fcol0_Transpose_D0_Block_col1.cwiseProduct(Fdotcol1_Transpose_D0_Block_col1);
    SPK2_5_D0_BlockDamping += Fcol0_Transpose_D0_Block_col2.cwiseProduct(Fdotcol1_Transpose_D0_Block_col2);
    m_SPK2_5_D0_Block += m_Alpha * SPK2_5_D0_BlockDamping;
    m_SPK2_5_D0_Block.array() *= m_GQWeight_det_J_0xi_D0.array();
    m_SPK2_5_D0_Block *= D0(5);

    ChMatrixNMc<double, 60, 3> P_transpose_scaled_Block_col_ordered; //1st tensor Piola-Kirchoff stress tensor (non-symmetric tensor) - Tiled across all Gauss-Quadrature points in column order in a big matrix

    P_transpose_scaled_Block_col_ordered.block<16, 1>(0, 0) = Fcol0_Transpose_D0_Block_col0.cwiseProduct(m_SPK2_0_D0_Block)
        + Fcol1_Transpose_D0_Block_col0.cwiseProduct(m_SPK2_5_D0_Block);
    P_transpose_scaled_Block_col_ordered.block<16, 1>(0, 0) += Fcol2_Transpose_D0_Block_col0.cwiseProduct(m_SPK2_4_D0_Block);
    P_transpose_scaled_Block_col_ordered.block<16, 1>(0, 1) = Fcol0_Transpose_D0_Block_col1.cwiseProduct(m_SPK2_0_D0_Block)
        + Fcol1_Transpose_D0_Block_col1.cwiseProduct(m_SPK2_5_D0_Block);
    P_transpose_scaled_Block_col_ordered.block<16, 1>(0, 1) += Fcol2_Transpose_D0_Block_col1.cwiseProduct(m_SPK2_4_D0_Block);
    P_transpose_scaled_Block_col_ordered.block<16, 1>(0, 2) = Fcol0_Transpose_D0_Block_col2.cwiseProduct(m_SPK2_0_D0_Block)
        + Fcol1_Transpose_D0_Block_col2.cwiseProduct(m_SPK2_5_D0_Block);
    P_transpose_scaled_Block_col_ordered.block<16, 1>(0, 2) += Fcol2_Transpose_D0_Block_col2.cwiseProduct(m_SPK2_4_D0_Block);

    P_transpose_scaled_Block_col_ordered.block<16, 1>(16, 0) = Fcol0_Transpose_D0_Block_col0.cwiseProduct(m_SPK2_5_D0_Block)
        + Fcol1_Transpose_D0_Block_col0.cwiseProduct(m_SPK2_1_D0_Block);
    P_transpose_scaled_Block_col_ordered.block<16, 1>(16, 0) += Fcol2_Transpose_D0_Block_col0.cwiseProduct(m_SPK2_3_D0_Block);
    P_transpose_scaled_Block_col_ordered.block<16, 1>(16, 1) = Fcol0_Transpose_D0_Block_col1.cwiseProduct(m_SPK2_5_D0_Block)
        + Fcol1_Transpose_D0_Block_col1.cwiseProduct(m_SPK2_1_D0_Block);
    P_transpose_scaled_Block_col_ordered.block<16, 1>(16, 1) += Fcol2_Transpose_D0_Block_col1.cwiseProduct(m_SPK2_3_D0_Block);
    P_transpose_scaled_Block_col_ordered.block<16, 1>(16, 2) = Fcol0_Transpose_D0_Block_col2.cwiseProduct(m_SPK2_5_D0_Block)
        + Fcol1_Transpose_D0_Block_col2.cwiseProduct(m_SPK2_1_D0_Block);
    P_transpose_scaled_Block_col_ordered.block<16, 1>(16, 2) += Fcol2_Transpose_D0_Block_col2.cwiseProduct(m_SPK2_3_D0_Block);

    P_transpose_scaled_Block_col_ordered.block<16, 1>(32, 0) = Fcol0_Transpose_D0_Block_col0.cwiseProduct(m_SPK2_4_D0_Block)
        + Fcol1_Transpose_D0_Block_col0.cwiseProduct(m_SPK2_3_D0_Block);
    P_transpose_scaled_Block_col_ordered.block<16, 1>(32, 0) += Fcol2_Transpose_D0_Block_col0.cwiseProduct(m_SPK2_2_D0_Block);
    P_transpose_scaled_Block_col_ordered.block<16, 1>(32, 1) = Fcol0_Transpose_D0_Block_col1.cwiseProduct(m_SPK2_4_D0_Block)
        + Fcol1_Transpose_D0_Block_col1.cwiseProduct(m_SPK2_3_D0_Block);
    P_transpose_scaled_Block_col_ordered.block<16, 1>(32, 1) += Fcol2_Transpose_D0_Block_col1.cwiseProduct(m_SPK2_2_D0_Block);
    P_transpose_scaled_Block_col_ordered.block<16, 1>(32, 2) = Fcol0_Transpose_D0_Block_col2.cwiseProduct(m_SPK2_4_D0_Block)
        + Fcol1_Transpose_D0_Block_col2.cwiseProduct(m_SPK2_3_D0_Block);
    P_transpose_scaled_Block_col_ordered.block<16, 1>(32, 2) += Fcol2_Transpose_D0_Block_col2.cwiseProduct(m_SPK2_2_D0_Block);

    //ChVectorN<double, 16> SPK2_0_D0_Block = D0(0)*m_GQWeight_det_J_0xi_D0.array()*(0.5*Fcol0_Transpose_D0_Block.rowwise().squaredNorm().array() - 0.5 + m_Alpha * Fcol0_Transpose_D0_Block.cwiseProduct(Fdotcol0_Transpose_D0_Block).rowwise().sum().array());
    //ChVectorN<double, 16> SPK2_1_D0_Block = D0(1)*m_GQWeight_det_J_0xi_D0.array()*(0.5*Fcol1_Transpose_D0_Block.rowwise().squaredNorm().array() - 0.5 + m_Alpha * Fcol1_Transpose_D0_Block.cwiseProduct(Fdotcol1_Transpose_D0_Block).rowwise().sum().array());
    //ChVectorN<double, 16> SPK2_2_D0_Block = D0(2)*m_GQWeight_det_J_0xi_D0.array()*(0.5*Fcol2_Transpose_D0_Block.rowwise().squaredNorm().array() - 0.5 + m_Alpha * Fcol2_Transpose_D0_Block.cwiseProduct(Fdotcol2_Transpose_D0_Block).rowwise().sum().array());
    //ChVectorN<double, 16> SPK2_3_D0_Block = D0(3)*m_GQWeight_det_J_0xi_D0.cwiseProduct(Fcol1_Transpose_D0_Block.cwiseProduct(Fcol2_Transpose_D0_Block).rowwise().sum() + m_Alpha * (Fcol1_Transpose_D0_Block.cwiseProduct(Fdotcol2_Transpose_D0_Block).rowwise().sum() + Fcol2_Transpose_D0_Block.cwiseProduct(Fdotcol1_Transpose_D0_Block).rowwise().sum()));
    //ChVectorN<double, 16> SPK2_4_D0_Block = D0(4)*m_GQWeight_det_J_0xi_D0.cwiseProduct(Fcol0_Transpose_D0_Block.cwiseProduct(Fcol2_Transpose_D0_Block).rowwise().sum() + m_Alpha * (Fcol0_Transpose_D0_Block.cwiseProduct(Fdotcol2_Transpose_D0_Block).rowwise().sum() + Fcol2_Transpose_D0_Block.cwiseProduct(Fdotcol0_Transpose_D0_Block).rowwise().sum()));
    //ChVectorN<double, 16> SPK2_5_D0_Block = D0(5)*m_GQWeight_det_J_0xi_D0.cwiseProduct(Fcol0_Transpose_D0_Block.cwiseProduct(Fcol1_Transpose_D0_Block).rowwise().sum() + m_Alpha * (Fcol0_Transpose_D0_Block.cwiseProduct(Fdotcol1_Transpose_D0_Block).rowwise().sum() + Fcol1_Transpose_D0_Block.cwiseProduct(Fdotcol0_Transpose_D0_Block).rowwise().sum()));

    //ChMatrixNM<double, 16, 3> Ptransposed_scaled_row0_D0_Block =
    //    Fcol0_Transpose_D0_Block.array().colwise()*SPK2_0_D0_Block.array() +
    //    Fcol1_Transpose_D0_Block.array().colwise()*SPK2_5_D0_Block.array() +
    //    Fcol2_Transpose_D0_Block.array().colwise()*SPK2_4_D0_Block.array();

    //ChMatrixNM<double, 16, 3> Ptransposed_scaled_row1_D0_Block =
    //    Fcol0_Transpose_D0_Block.array().colwise()*SPK2_5_D0_Block.array() +
    //    Fcol1_Transpose_D0_Block.array().colwise()*SPK2_1_D0_Block.array() +
    //    Fcol2_Transpose_D0_Block.array().colwise()*SPK2_3_D0_Block.array();

    //ChMatrixNM<double, 16, 3> Ptransposed_scaled_row2_D0_Block =
    //    Fcol0_Transpose_D0_Block.array().colwise()*SPK2_4_D0_Block.array() +
    //    Fcol1_Transpose_D0_Block.array().colwise()*SPK2_3_D0_Block.array() +
    //    Fcol2_Transpose_D0_Block.array().colwise()*SPK2_2_D0_Block.array();

    // =============================================================================

    ChVectorN<double, 4> Ediag_0_Dv_Block = Fcol0_Transpose_Dv_Block_col0.cwiseProduct(Fcol0_Transpose_Dv_Block_col0);
    Ediag_0_Dv_Block += Fcol0_Transpose_Dv_Block_col1.cwiseProduct(Fcol0_Transpose_Dv_Block_col1);
    Ediag_0_Dv_Block += Fcol0_Transpose_Dv_Block_col2.cwiseProduct(Fcol0_Transpose_Dv_Block_col2);
    Ediag_0_Dv_Block.array() -= 1;
    Ediag_0_Dv_Block *= 0.5;
    ChVectorN<double, 4> Ediag_0_Dv_BlockDamping = Fcol0_Transpose_Dv_Block_col0.cwiseProduct(Fdotcol0_Transpose_Dv_Block_col0);
    Ediag_0_Dv_BlockDamping += Fcol0_Transpose_Dv_Block_col1.cwiseProduct(Fdotcol0_Transpose_Dv_Block_col1);
    Ediag_0_Dv_BlockDamping += Fcol0_Transpose_Dv_Block_col2.cwiseProduct(Fdotcol0_Transpose_Dv_Block_col2);
    Ediag_0_Dv_Block += m_Alpha * Ediag_0_Dv_BlockDamping;
    Ediag_0_Dv_Block.array() *= m_GQWeight_det_J_0xi_Dv.array();

    ChVectorN<double, 4> Ediag_1_Dv_Block = Fcol1_Transpose_Dv_Block_col0.cwiseProduct(Fcol1_Transpose_Dv_Block_col0);
    Ediag_1_Dv_Block += Fcol1_Transpose_Dv_Block_col1.cwiseProduct(Fcol1_Transpose_Dv_Block_col1);
    Ediag_1_Dv_Block += Fcol1_Transpose_Dv_Block_col2.cwiseProduct(Fcol1_Transpose_Dv_Block_col2);
    Ediag_1_Dv_Block.array() -= 1;
    Ediag_1_Dv_Block *= 0.5;
    ChVectorN<double, 4> Ediag_1_Dv_BlockDamping = Fcol1_Transpose_Dv_Block_col0.cwiseProduct(Fdotcol1_Transpose_Dv_Block_col0);
    Ediag_1_Dv_BlockDamping += Fcol1_Transpose_Dv_Block_col1.cwiseProduct(Fdotcol1_Transpose_Dv_Block_col1);
    Ediag_1_Dv_BlockDamping += Fcol1_Transpose_Dv_Block_col2.cwiseProduct(Fdotcol1_Transpose_Dv_Block_col2);
    Ediag_1_Dv_Block += m_Alpha * Ediag_1_Dv_BlockDamping;
    Ediag_1_Dv_Block.array() *= m_GQWeight_det_J_0xi_Dv.array();

    ChVectorN<double, 4> Ediag_2_Dv_Block = Fcol2_Transpose_Dv_Block_col0.cwiseProduct(Fcol2_Transpose_Dv_Block_col0);
    Ediag_2_Dv_Block += Fcol2_Transpose_Dv_Block_col1.cwiseProduct(Fcol2_Transpose_Dv_Block_col1);
    Ediag_2_Dv_Block += Fcol2_Transpose_Dv_Block_col2.cwiseProduct(Fcol2_Transpose_Dv_Block_col2);
    Ediag_2_Dv_Block.array() -= 1;
    Ediag_2_Dv_Block *= 0.5;
    ChVectorN<double, 4> Ediag_2_Dv_BlockDamping = Fcol2_Transpose_Dv_Block_col0.cwiseProduct(Fdotcol2_Transpose_Dv_Block_col0);
    Ediag_2_Dv_BlockDamping += Fcol2_Transpose_Dv_Block_col1.cwiseProduct(Fdotcol2_Transpose_Dv_Block_col1);
    Ediag_2_Dv_BlockDamping += Fcol2_Transpose_Dv_Block_col2.cwiseProduct(Fdotcol2_Transpose_Dv_Block_col2);
    Ediag_2_Dv_Block += m_Alpha * Ediag_2_Dv_BlockDamping;
    Ediag_2_Dv_Block.array() *= m_GQWeight_det_J_0xi_Dv.array();

    //ChVectorN<double, 4> Ediag_0_Dv_Block = m_GQWeight_det_J_0xi_Dv.array() * (0.5*Fcol0_Transpose_Dv_Block.rowwise().squaredNorm().array() - 0.5 + m_Alpha * Fcol0_Transpose_Dv_Block.cwiseProduct(Fdotcol0_Transpose_Dv_Block).rowwise().sum().array());
    //ChVectorN<double, 4> Ediag_1_Dv_Block = m_GQWeight_det_J_0xi_Dv.array() * (0.5*Fcol1_Transpose_Dv_Block.rowwise().squaredNorm().array() - 0.5 + m_Alpha * Fcol1_Transpose_Dv_Block.cwiseProduct(Fdotcol1_Transpose_Dv_Block).rowwise().sum().array());
    //ChVectorN<double, 4> Ediag_2_Dv_Block = m_GQWeight_det_J_0xi_Dv.array() * (0.5*Fcol2_Transpose_Dv_Block.rowwise().squaredNorm().array() - 0.5 + m_Alpha * Fcol2_Transpose_Dv_Block.cwiseProduct(Fdotcol2_Transpose_Dv_Block).rowwise().sum().array());

    ChMatrix33<double> Dv = GetMaterial()->Get_Dv();

    m_Sdiag_0_Dv_Block.noalias() = Dv(0, 0)*Ediag_0_Dv_Block + Dv(1, 0)*Ediag_1_Dv_Block + Dv(2, 0)*Ediag_2_Dv_Block;
    m_Sdiag_1_Dv_Block.noalias() = Dv(0, 1)*Ediag_0_Dv_Block + Dv(1, 1)*Ediag_1_Dv_Block + Dv(2, 1)*Ediag_2_Dv_Block;
    m_Sdiag_2_Dv_Block.noalias() = Dv(0, 2)*Ediag_0_Dv_Block + Dv(1, 2)*Ediag_1_Dv_Block + Dv(2, 2)*Ediag_2_Dv_Block;


    P_transpose_scaled_Block_col_ordered.block<4, 1>(48, 0) = Fcol0_Transpose_Dv_Block_col0.cwiseProduct(m_Sdiag_0_Dv_Block);
    P_transpose_scaled_Block_col_ordered.block<4, 1>(48, 1) = Fcol0_Transpose_Dv_Block_col1.cwiseProduct(m_Sdiag_0_Dv_Block);
    P_transpose_scaled_Block_col_ordered.block<4, 1>(48, 2) = Fcol0_Transpose_Dv_Block_col2.cwiseProduct(m_Sdiag_0_Dv_Block);

    P_transpose_scaled_Block_col_ordered.block<4, 1>(52, 0) = Fcol1_Transpose_Dv_Block_col0.cwiseProduct(m_Sdiag_1_Dv_Block);
    P_transpose_scaled_Block_col_ordered.block<4, 1>(52, 1) = Fcol1_Transpose_Dv_Block_col1.cwiseProduct(m_Sdiag_1_Dv_Block);
    P_transpose_scaled_Block_col_ordered.block<4, 1>(52, 2) = Fcol1_Transpose_Dv_Block_col2.cwiseProduct(m_Sdiag_1_Dv_Block);

    P_transpose_scaled_Block_col_ordered.block<4, 1>(56, 0) = Fcol2_Transpose_Dv_Block_col0.cwiseProduct(m_Sdiag_2_Dv_Block);
    P_transpose_scaled_Block_col_ordered.block<4, 1>(56, 1) = Fcol2_Transpose_Dv_Block_col1.cwiseProduct(m_Sdiag_2_Dv_Block);
    P_transpose_scaled_Block_col_ordered.block<4, 1>(56, 2) = Fcol2_Transpose_Dv_Block_col2.cwiseProduct(m_Sdiag_2_Dv_Block);

    //ChMatrixNM<double, 4, 3> Ptransposed_scaled_row0_Dv_Block =
    //    Fcol0_Transpose_Dv_Block.array().colwise()*Sdiag_0_Dv_Block.array();
    //ChMatrixNM<double, 4, 3> Ptransposed_scaled_row1_Dv_Block =
    //    Fcol1_Transpose_Dv_Block.array().colwise()*Sdiag_1_Dv_Block.array();
    //ChMatrixNM<double, 4, 3> Ptransposed_scaled_row2_Dv_Block =
    //    Fcol2_Transpose_Dv_Block.array().colwise()*Sdiag_2_Dv_Block.array();

    // =============================================================================

    //ChMatrixNM<double, 16, 3> Ptransposed_scaled_row0_D0_Block;
    //Ptransposed_scaled_row0_D0_Block.col(0) = Fcol0_Transpose_D0_Block.col(0).cwiseProduct(SPK2_0_D0_Block);
    //Ptransposed_scaled_row0_D0_Block.col(0) += Fcol1_Transpose_D0_Block.col(0).cwiseProduct(SPK2_5_D0_Block);
    //Ptransposed_scaled_row0_D0_Block.col(0) += Fcol2_Transpose_D0_Block.col(0).cwiseProduct(SPK2_4_D0_Block);
    //Ptransposed_scaled_row0_D0_Block.col(1) = Fcol0_Transpose_D0_Block.col(1).cwiseProduct(SPK2_0_D0_Block);
    //Ptransposed_scaled_row0_D0_Block.col(1) += Fcol1_Transpose_D0_Block.col(1).cwiseProduct(SPK2_5_D0_Block);
    //Ptransposed_scaled_row0_D0_Block.col(1) += Fcol2_Transpose_D0_Block.col(1).cwiseProduct(SPK2_4_D0_Block);
    //Ptransposed_scaled_row0_D0_Block.col(2) = Fcol0_Transpose_D0_Block.col(2).cwiseProduct(SPK2_0_D0_Block);
    //Ptransposed_scaled_row0_D0_Block.col(2) += Fcol1_Transpose_D0_Block.col(2).cwiseProduct(SPK2_5_D0_Block);
    //Ptransposed_scaled_row0_D0_Block.col(2) += Fcol2_Transpose_D0_Block.col(2).cwiseProduct(SPK2_4_D0_Block);

    //ChMatrixNM<double, 16, 3> Ptransposed_scaled_row1_D0_Block;
    //Ptransposed_scaled_row1_D0_Block.col(0) = Fcol0_Transpose_D0_Block.col(0).cwiseProduct(SPK2_5_D0_Block);
    //Ptransposed_scaled_row1_D0_Block.col(0) += Fcol1_Transpose_D0_Block.col(0).cwiseProduct(SPK2_1_D0_Block);
    //Ptransposed_scaled_row1_D0_Block.col(0) += Fcol2_Transpose_D0_Block.col(0).cwiseProduct(SPK2_3_D0_Block);
    //Ptransposed_scaled_row1_D0_Block.col(1) = Fcol0_Transpose_D0_Block.col(1).cwiseProduct(SPK2_5_D0_Block);
    //Ptransposed_scaled_row1_D0_Block.col(1) += Fcol1_Transpose_D0_Block.col(1).cwiseProduct(SPK2_1_D0_Block);
    //Ptransposed_scaled_row1_D0_Block.col(1) += Fcol2_Transpose_D0_Block.col(1).cwiseProduct(SPK2_3_D0_Block);
    //Ptransposed_scaled_row1_D0_Block.col(2) = Fcol0_Transpose_D0_Block.col(2).cwiseProduct(SPK2_5_D0_Block);
    //Ptransposed_scaled_row1_D0_Block.col(2) += Fcol1_Transpose_D0_Block.col(2).cwiseProduct(SPK2_1_D0_Block);
    //Ptransposed_scaled_row1_D0_Block.col(2) += Fcol2_Transpose_D0_Block.col(2).cwiseProduct(SPK2_3_D0_Block);

    //ChMatrixNM<double, 16, 3> Ptransposed_scaled_row2_D0_Block;
    //Ptransposed_scaled_row2_D0_Block.col(0) = Fcol0_Transpose_D0_Block.col(0).cwiseProduct(SPK2_4_D0_Block);
    //Ptransposed_scaled_row2_D0_Block.col(0) += Fcol1_Transpose_D0_Block.col(0).cwiseProduct(SPK2_3_D0_Block);
    //Ptransposed_scaled_row2_D0_Block.col(0) += Fcol2_Transpose_D0_Block.col(0).cwiseProduct(SPK2_2_D0_Block);
    //Ptransposed_scaled_row2_D0_Block.col(1) = Fcol0_Transpose_D0_Block.col(1).cwiseProduct(SPK2_4_D0_Block);
    //Ptransposed_scaled_row2_D0_Block.col(1) += Fcol1_Transpose_D0_Block.col(1).cwiseProduct(SPK2_3_D0_Block);
    //Ptransposed_scaled_row2_D0_Block.col(1) += Fcol2_Transpose_D0_Block.col(1).cwiseProduct(SPK2_2_D0_Block);
    //Ptransposed_scaled_row2_D0_Block.col(2) = Fcol0_Transpose_D0_Block.col(2).cwiseProduct(SPK2_4_D0_Block);
    //Ptransposed_scaled_row2_D0_Block.col(2) += Fcol1_Transpose_D0_Block.col(2).cwiseProduct(SPK2_3_D0_Block);
    //Ptransposed_scaled_row2_D0_Block.col(2) += Fcol2_Transpose_D0_Block.col(2).cwiseProduct(SPK2_2_D0_Block);

    //ChMatrixNM<double, 4, 3> Ptransposed_scaled_row0_Dv_Block;
    //Ptransposed_scaled_row0_Dv_Block.col(0) = Fcol0_Transpose_Dv_Block.col(0).cwiseProduct(Sdiag_0_Dv_Block);
    //Ptransposed_scaled_row0_Dv_Block.col(1) = Fcol0_Transpose_Dv_Block.col(1).cwiseProduct(Sdiag_0_Dv_Block);
    //Ptransposed_scaled_row0_Dv_Block.col(2) = Fcol0_Transpose_Dv_Block.col(2).cwiseProduct(Sdiag_0_Dv_Block);

    //ChMatrixNM<double, 4, 3> Ptransposed_scaled_row1_Dv_Block;
    //Ptransposed_scaled_row1_Dv_Block.col(0) = Fcol1_Transpose_Dv_Block.col(0).cwiseProduct(Sdiag_1_Dv_Block);
    //Ptransposed_scaled_row1_Dv_Block.col(1) = Fcol1_Transpose_Dv_Block.col(1).cwiseProduct(Sdiag_1_Dv_Block);
    //Ptransposed_scaled_row1_Dv_Block.col(2) = Fcol1_Transpose_Dv_Block.col(2).cwiseProduct(Sdiag_1_Dv_Block);

    //ChMatrixNM<double, 4, 3> Ptransposed_scaled_row2_Dv_Block;
    //Ptransposed_scaled_row2_Dv_Block.col(0) = Fcol2_Transpose_Dv_Block.col(0).cwiseProduct(Sdiag_2_Dv_Block);
    //Ptransposed_scaled_row2_Dv_Block.col(1) = Fcol2_Transpose_Dv_Block.col(1).cwiseProduct(Sdiag_2_Dv_Block);
    //Ptransposed_scaled_row2_Dv_Block.col(2) = Fcol2_Transpose_Dv_Block.col(2).cwiseProduct(Sdiag_2_Dv_Block);

    //ChMatrixNM<double, 9, 3> QiCompact =
    //    m_SD_precompute_D0_col0_block * Ptransposed_scaled_row0_D0_Block +
    //    m_SD_precompute_D0_col1_block * Ptransposed_scaled_row1_D0_Block +
    //    m_SD_precompute_D0_col2_block * Ptransposed_scaled_row2_D0_Block +
    //    m_SD_precompute_Dv_col0_block * Ptransposed_scaled_row0_Dv_Block +
    //    m_SD_precompute_Dv_col1_block * Ptransposed_scaled_row1_Dv_Block +
    //    m_SD_precompute_Dv_col2_block * Ptransposed_scaled_row2_Dv_Block;

    ChMatrixNM<double, 9, 3> QiCompact = m_SD_precompute_col_ordered * P_transpose_scaled_Block_col_ordered;
    Eigen::Map<ChVectorN<double, 27>> QiReshaped(QiCompact.data(), QiCompact.size());
    Fi = QiReshaped;
}

void ChElementBeamANCF_MT35::ComputeInternalForcesAtStateNoDamping(ChVectorDynamic<>& Fi, const ChMatrixNMc<double, 9, 3>& e_bar) {

    // Straight & Normalized Internal Force Integrand is of order : 8 in xi, order : 4 in eta, and order : 4 in zeta.
    // This requires GQ 5 points along the xi direction and 3 points along the eta and zeta directions for "Full Integration"
    // However, very similar results can be obtained with 1 fewer GQ point in  each direction, resulting in roughly 1/3 of the calculations

    //ChMatrixNMc<double, 16, 3> Fcol0_Transpose_D0_Block = m_SD_precompute_D0_col0_block.transpose()*e_bar;
    //ChMatrixNMc<double, 16, 3> Fcol1_Transpose_D0_Block = m_SD_precompute_D0_col1_block.transpose()*e_bar;
    //ChMatrixNMc<double, 16, 3> Fcol2_Transpose_D0_Block = m_SD_precompute_D0_col2_block.transpose()*e_bar;

    //Calculate F is one big block and then split up afterwards to improve efficiency (hopefully)
    m_F_Transpose_CombinedBlock_col_ordered = m_SD_precompute_col_ordered.transpose()*e_bar;

    ChMatrixNMc<double, 16, 1> Fcol0_Transpose_D0_Block_col0 = m_F_Transpose_CombinedBlock_col_ordered.block<16, 1>(0, 0);
    ChMatrixNMc<double, 16, 1> Fcol0_Transpose_D0_Block_col1 = m_F_Transpose_CombinedBlock_col_ordered.block<16, 1>(0, 1);
    ChMatrixNMc<double, 16, 1> Fcol0_Transpose_D0_Block_col2 = m_F_Transpose_CombinedBlock_col_ordered.block<16, 1>(0, 2);

    ChMatrixNMc<double, 16, 1> Fcol1_Transpose_D0_Block_col0 = m_F_Transpose_CombinedBlock_col_ordered.block<16, 1>(16, 0);
    ChMatrixNMc<double, 16, 1> Fcol1_Transpose_D0_Block_col1 = m_F_Transpose_CombinedBlock_col_ordered.block<16, 1>(16, 1);
    ChMatrixNMc<double, 16, 1> Fcol1_Transpose_D0_Block_col2 = m_F_Transpose_CombinedBlock_col_ordered.block<16, 1>(16, 2);

    ChMatrixNMc<double, 16, 1> Fcol2_Transpose_D0_Block_col0 = m_F_Transpose_CombinedBlock_col_ordered.block<16, 1>(32, 0);
    ChMatrixNMc<double, 16, 1> Fcol2_Transpose_D0_Block_col1 = m_F_Transpose_CombinedBlock_col_ordered.block<16, 1>(32, 1);
    ChMatrixNMc<double, 16, 1> Fcol2_Transpose_D0_Block_col2 = m_F_Transpose_CombinedBlock_col_ordered.block<16, 1>(32, 2);

    ChMatrixNMc<double, 4, 1> Fcol0_Transpose_Dv_Block_col0 = m_F_Transpose_CombinedBlock_col_ordered.block<4, 1>(48, 0);
    ChMatrixNMc<double, 4, 1> Fcol0_Transpose_Dv_Block_col1 = m_F_Transpose_CombinedBlock_col_ordered.block<4, 1>(48, 1);
    ChMatrixNMc<double, 4, 1> Fcol0_Transpose_Dv_Block_col2 = m_F_Transpose_CombinedBlock_col_ordered.block<4, 1>(48, 2);

    ChMatrixNMc<double, 4, 1> Fcol1_Transpose_Dv_Block_col0 = m_F_Transpose_CombinedBlock_col_ordered.block<4, 1>(52, 0);
    ChMatrixNMc<double, 4, 1> Fcol1_Transpose_Dv_Block_col1 = m_F_Transpose_CombinedBlock_col_ordered.block<4, 1>(52, 1);
    ChMatrixNMc<double, 4, 1> Fcol1_Transpose_Dv_Block_col2 = m_F_Transpose_CombinedBlock_col_ordered.block<4, 1>(52, 2);

    ChMatrixNMc<double, 4, 1> Fcol2_Transpose_Dv_Block_col0 = m_F_Transpose_CombinedBlock_col_ordered.block<4, 1>(56, 0);
    ChMatrixNMc<double, 4, 1> Fcol2_Transpose_Dv_Block_col1 = m_F_Transpose_CombinedBlock_col_ordered.block<4, 1>(56, 1);
    ChMatrixNMc<double, 4, 1> Fcol2_Transpose_Dv_Block_col2 = m_F_Transpose_CombinedBlock_col_ordered.block<4, 1>(56, 2);

    ChVectorN<double, 6> D0 = GetMaterial()->Get_D0();

    //ChVectorN<double, 16> SPK2_0_D0_Block = Fcol0_Transpose_D0_Block.col(0).cwiseProduct(Fcol0_Transpose_D0_Block.col(0));
    //SPK2_0_D0_Block += Fcol0_Transpose_D0_Block.col(1).cwiseProduct(Fcol0_Transpose_D0_Block.col(1));
    //SPK2_0_D0_Block += Fcol0_Transpose_D0_Block.col(2).cwiseProduct(Fcol0_Transpose_D0_Block.col(2));
    //SPK2_0_D0_Block.array() -= 1;
    //SPK2_0_D0_Block.array() *= 0.5*D0(0)*m_GQWeight_det_J_0xi_D0.array();

    //ChVectorN<double, 16> SPK2_1_D0_Block = Fcol1_Transpose_D0_Block.col(0).cwiseProduct(Fcol1_Transpose_D0_Block.col(0));
    //SPK2_1_D0_Block += Fcol1_Transpose_D0_Block.col(1).cwiseProduct(Fcol1_Transpose_D0_Block.col(1));
    //SPK2_1_D0_Block += Fcol1_Transpose_D0_Block.col(2).cwiseProduct(Fcol1_Transpose_D0_Block.col(2));
    //SPK2_1_D0_Block.array() -= 1;
    //SPK2_1_D0_Block.array() *= 0.5*D0(1)*m_GQWeight_det_J_0xi_D0.array();

    //ChVectorN<double, 16> SPK2_2_D0_Block = Fcol2_Transpose_D0_Block.col(0).cwiseProduct(Fcol2_Transpose_D0_Block.col(0));
    //SPK2_2_D0_Block += Fcol2_Transpose_D0_Block.col(1).cwiseProduct(Fcol2_Transpose_D0_Block.col(1));
    //SPK2_2_D0_Block += Fcol2_Transpose_D0_Block.col(2).cwiseProduct(Fcol2_Transpose_D0_Block.col(2));
    //SPK2_2_D0_Block.array() -= 1;
    //SPK2_2_D0_Block.array() *= 0.5*D0(2)*m_GQWeight_det_J_0xi_D0.array();

    //ChVectorN<double, 16> SPK2_3_D0_Block = Fcol1_Transpose_D0_Block.col(0).cwiseProduct(Fcol2_Transpose_D0_Block.col(0));
    //SPK2_3_D0_Block += Fcol1_Transpose_D0_Block.col(1).cwiseProduct(Fcol2_Transpose_D0_Block.col(1));
    //SPK2_3_D0_Block += Fcol1_Transpose_D0_Block.col(2).cwiseProduct(Fcol2_Transpose_D0_Block.col(2));
    //SPK2_3_D0_Block.array() *= D0(3)*m_GQWeight_det_J_0xi_D0.array();

    //ChVectorN<double, 16> SPK2_4_D0_Block = Fcol0_Transpose_D0_Block.col(0).cwiseProduct(Fcol2_Transpose_D0_Block.col(0));
    //SPK2_4_D0_Block += Fcol0_Transpose_D0_Block.col(1).cwiseProduct(Fcol2_Transpose_D0_Block.col(1));
    //SPK2_4_D0_Block += Fcol0_Transpose_D0_Block.col(2).cwiseProduct(Fcol2_Transpose_D0_Block.col(2));
    //SPK2_4_D0_Block.array() *= D0(4)*m_GQWeight_det_J_0xi_D0.array();

    //ChVectorN<double, 16> SPK2_5_D0_Block = Fcol0_Transpose_D0_Block.col(0).cwiseProduct(Fcol1_Transpose_D0_Block.col(0));
    //SPK2_5_D0_Block += Fcol0_Transpose_D0_Block.col(1).cwiseProduct(Fcol1_Transpose_D0_Block.col(1));
    //SPK2_5_D0_Block += Fcol0_Transpose_D0_Block.col(2).cwiseProduct(Fcol1_Transpose_D0_Block.col(2));
    //SPK2_5_D0_Block.array() *= D0(5)*m_GQWeight_det_J_0xi_D0.array();

    m_SPK2_0_D0_Block = Fcol0_Transpose_D0_Block_col0.cwiseProduct(Fcol0_Transpose_D0_Block_col0);
    m_SPK2_0_D0_Block += Fcol0_Transpose_D0_Block_col1.cwiseProduct(Fcol0_Transpose_D0_Block_col1);
    m_SPK2_0_D0_Block += Fcol0_Transpose_D0_Block_col2.cwiseProduct(Fcol0_Transpose_D0_Block_col2);
    m_SPK2_0_D0_Block.array() -= 1;
    m_SPK2_0_D0_Block.array() *= m_GQWeight_det_J_0xi_D0.array();
    m_SPK2_0_D0_Block *= (0.5 * D0(0));

    m_SPK2_1_D0_Block = Fcol1_Transpose_D0_Block_col0.cwiseProduct(Fcol1_Transpose_D0_Block_col0);
    m_SPK2_1_D0_Block += Fcol1_Transpose_D0_Block_col1.cwiseProduct(Fcol1_Transpose_D0_Block_col1);
    m_SPK2_1_D0_Block += Fcol1_Transpose_D0_Block_col2.cwiseProduct(Fcol1_Transpose_D0_Block_col2);
    m_SPK2_1_D0_Block.array() -= 1;
    m_SPK2_1_D0_Block.array() *= m_GQWeight_det_J_0xi_D0.array();
    m_SPK2_1_D0_Block *= (0.5 * D0(1));

    m_SPK2_2_D0_Block = Fcol2_Transpose_D0_Block_col0.cwiseProduct(Fcol2_Transpose_D0_Block_col0);
    m_SPK2_2_D0_Block += Fcol2_Transpose_D0_Block_col1.cwiseProduct(Fcol2_Transpose_D0_Block_col1);
    m_SPK2_2_D0_Block += Fcol2_Transpose_D0_Block_col2.cwiseProduct(Fcol2_Transpose_D0_Block_col2);
    m_SPK2_2_D0_Block.array() -= 1;
    m_SPK2_2_D0_Block.array() *= m_GQWeight_det_J_0xi_D0.array();
    m_SPK2_2_D0_Block *= (0.5 * D0(2));

    m_SPK2_3_D0_Block = Fcol1_Transpose_D0_Block_col0.cwiseProduct(Fcol2_Transpose_D0_Block_col0);
    m_SPK2_3_D0_Block += Fcol1_Transpose_D0_Block_col1.cwiseProduct(Fcol2_Transpose_D0_Block_col1);
    m_SPK2_3_D0_Block += Fcol1_Transpose_D0_Block_col2.cwiseProduct(Fcol2_Transpose_D0_Block_col2);
    m_SPK2_3_D0_Block.array() *= m_GQWeight_det_J_0xi_D0.array();
    m_SPK2_3_D0_Block *= D0(3);

    m_SPK2_4_D0_Block = Fcol0_Transpose_D0_Block_col0.cwiseProduct(Fcol2_Transpose_D0_Block_col0);
    m_SPK2_4_D0_Block += Fcol0_Transpose_D0_Block_col1.cwiseProduct(Fcol2_Transpose_D0_Block_col1);
    m_SPK2_4_D0_Block += Fcol0_Transpose_D0_Block_col2.cwiseProduct(Fcol2_Transpose_D0_Block_col2);
    m_SPK2_4_D0_Block.array() *= m_GQWeight_det_J_0xi_D0.array();
    m_SPK2_4_D0_Block *= D0(4);

    m_SPK2_5_D0_Block = Fcol0_Transpose_D0_Block_col0.cwiseProduct(Fcol1_Transpose_D0_Block_col0);
    m_SPK2_5_D0_Block += Fcol0_Transpose_D0_Block_col1.cwiseProduct(Fcol1_Transpose_D0_Block_col1);
    m_SPK2_5_D0_Block += Fcol0_Transpose_D0_Block_col2.cwiseProduct(Fcol1_Transpose_D0_Block_col2);
    m_SPK2_5_D0_Block.array() *= m_GQWeight_det_J_0xi_D0.array();
    m_SPK2_5_D0_Block *= D0(5);

    ChMatrixNMc<double, 60, 3> P_transpose_scaled_Block_col_ordered; //1st tensor Piola-Kirchoff stress tensor (non-symmetric tensor) - Tiled across all Gauss-Quadrature points in column order in a big matrix

    P_transpose_scaled_Block_col_ordered.block<16, 1>(0, 0) = Fcol0_Transpose_D0_Block_col0.cwiseProduct(m_SPK2_0_D0_Block)
        + Fcol1_Transpose_D0_Block_col0.cwiseProduct(m_SPK2_5_D0_Block);
    P_transpose_scaled_Block_col_ordered.block<16, 1>(0, 0) += Fcol2_Transpose_D0_Block_col0.cwiseProduct(m_SPK2_4_D0_Block);
    P_transpose_scaled_Block_col_ordered.block<16, 1>(0, 1) = Fcol0_Transpose_D0_Block_col1.cwiseProduct(m_SPK2_0_D0_Block)
        + Fcol1_Transpose_D0_Block_col1.cwiseProduct(m_SPK2_5_D0_Block);
    P_transpose_scaled_Block_col_ordered.block<16, 1>(0, 1) += Fcol2_Transpose_D0_Block_col1.cwiseProduct(m_SPK2_4_D0_Block);
    P_transpose_scaled_Block_col_ordered.block<16, 1>(0, 2) = Fcol0_Transpose_D0_Block_col2.cwiseProduct(m_SPK2_0_D0_Block)
        + Fcol1_Transpose_D0_Block_col2.cwiseProduct(m_SPK2_5_D0_Block);
    P_transpose_scaled_Block_col_ordered.block<16, 1>(0, 2) += Fcol2_Transpose_D0_Block_col2.cwiseProduct(m_SPK2_4_D0_Block);

    P_transpose_scaled_Block_col_ordered.block<16, 1>(16, 0) = Fcol0_Transpose_D0_Block_col0.cwiseProduct(m_SPK2_5_D0_Block)
        + Fcol1_Transpose_D0_Block_col0.cwiseProduct(m_SPK2_1_D0_Block);
    P_transpose_scaled_Block_col_ordered.block<16, 1>(16, 0) += Fcol2_Transpose_D0_Block_col0.cwiseProduct(m_SPK2_3_D0_Block);
    P_transpose_scaled_Block_col_ordered.block<16, 1>(16, 1) = Fcol0_Transpose_D0_Block_col1.cwiseProduct(m_SPK2_5_D0_Block)
        + Fcol1_Transpose_D0_Block_col1.cwiseProduct(m_SPK2_1_D0_Block);
    P_transpose_scaled_Block_col_ordered.block<16, 1>(16, 1) += Fcol2_Transpose_D0_Block_col1.cwiseProduct(m_SPK2_3_D0_Block);
    P_transpose_scaled_Block_col_ordered.block<16, 1>(16, 2) = Fcol0_Transpose_D0_Block_col2.cwiseProduct(m_SPK2_5_D0_Block)
        + Fcol1_Transpose_D0_Block_col2.cwiseProduct(m_SPK2_1_D0_Block);
    P_transpose_scaled_Block_col_ordered.block<16, 1>(16, 2) += Fcol2_Transpose_D0_Block_col2.cwiseProduct(m_SPK2_3_D0_Block);

    P_transpose_scaled_Block_col_ordered.block<16, 1>(32, 0) = Fcol0_Transpose_D0_Block_col0.cwiseProduct(m_SPK2_4_D0_Block)
        + Fcol1_Transpose_D0_Block_col0.cwiseProduct(m_SPK2_3_D0_Block);
    P_transpose_scaled_Block_col_ordered.block<16, 1>(32, 0) += Fcol2_Transpose_D0_Block_col0.cwiseProduct(m_SPK2_2_D0_Block);
    P_transpose_scaled_Block_col_ordered.block<16, 1>(32, 1) = Fcol0_Transpose_D0_Block_col1.cwiseProduct(m_SPK2_4_D0_Block)
        + Fcol1_Transpose_D0_Block_col1.cwiseProduct(m_SPK2_3_D0_Block);
    P_transpose_scaled_Block_col_ordered.block<16, 1>(32, 1) += Fcol2_Transpose_D0_Block_col1.cwiseProduct(m_SPK2_2_D0_Block);
    P_transpose_scaled_Block_col_ordered.block<16, 1>(32, 2) = Fcol0_Transpose_D0_Block_col2.cwiseProduct(m_SPK2_4_D0_Block)
        + Fcol1_Transpose_D0_Block_col2.cwiseProduct(m_SPK2_3_D0_Block);
    P_transpose_scaled_Block_col_ordered.block<16, 1>(32, 2) += Fcol2_Transpose_D0_Block_col2.cwiseProduct(m_SPK2_2_D0_Block);

    //ChMatrixNM<double, 16, 3> Ptransposed_scaled_row0_D0_Block;
    //Ptransposed_scaled_row0_D0_Block.col(0) = Fcol0_Transpose_D0_Block.col(0).cwiseProduct(SPK2_0_D0_Block);
    //Ptransposed_scaled_row0_D0_Block.col(0) += Fcol1_Transpose_D0_Block.col(0).cwiseProduct(SPK2_5_D0_Block);
    //Ptransposed_scaled_row0_D0_Block.col(0) += Fcol2_Transpose_D0_Block.col(0).cwiseProduct(SPK2_4_D0_Block);
    //Ptransposed_scaled_row0_D0_Block.col(1) = Fcol0_Transpose_D0_Block.col(1).cwiseProduct(SPK2_0_D0_Block);
    //Ptransposed_scaled_row0_D0_Block.col(1) += Fcol1_Transpose_D0_Block.col(1).cwiseProduct(SPK2_5_D0_Block);
    //Ptransposed_scaled_row0_D0_Block.col(1) += Fcol2_Transpose_D0_Block.col(1).cwiseProduct(SPK2_4_D0_Block);
    //Ptransposed_scaled_row0_D0_Block.col(2) = Fcol0_Transpose_D0_Block.col(2).cwiseProduct(SPK2_0_D0_Block);
    //Ptransposed_scaled_row0_D0_Block.col(2) += Fcol1_Transpose_D0_Block.col(2).cwiseProduct(SPK2_5_D0_Block);
    //Ptransposed_scaled_row0_D0_Block.col(2) += Fcol2_Transpose_D0_Block.col(2).cwiseProduct(SPK2_4_D0_Block);

    //ChMatrixNM<double, 16, 3> Ptransposed_scaled_row1_D0_Block;
    //Ptransposed_scaled_row1_D0_Block.col(0) = Fcol0_Transpose_D0_Block.col(0).cwiseProduct(SPK2_5_D0_Block);
    //Ptransposed_scaled_row1_D0_Block.col(0) += Fcol1_Transpose_D0_Block.col(0).cwiseProduct(SPK2_1_D0_Block);
    //Ptransposed_scaled_row1_D0_Block.col(0) += Fcol2_Transpose_D0_Block.col(0).cwiseProduct(SPK2_3_D0_Block);
    //Ptransposed_scaled_row1_D0_Block.col(1) = Fcol0_Transpose_D0_Block.col(1).cwiseProduct(SPK2_5_D0_Block);
    //Ptransposed_scaled_row1_D0_Block.col(1) += Fcol1_Transpose_D0_Block.col(1).cwiseProduct(SPK2_1_D0_Block);
    //Ptransposed_scaled_row1_D0_Block.col(1) += Fcol2_Transpose_D0_Block.col(1).cwiseProduct(SPK2_3_D0_Block);
    //Ptransposed_scaled_row1_D0_Block.col(2) = Fcol0_Transpose_D0_Block.col(2).cwiseProduct(SPK2_5_D0_Block);
    //Ptransposed_scaled_row1_D0_Block.col(2) += Fcol1_Transpose_D0_Block.col(2).cwiseProduct(SPK2_1_D0_Block);
    //Ptransposed_scaled_row1_D0_Block.col(2) += Fcol2_Transpose_D0_Block.col(2).cwiseProduct(SPK2_3_D0_Block);

    //ChMatrixNM<double, 16, 3> Ptransposed_scaled_row2_D0_Block;
    //Ptransposed_scaled_row2_D0_Block.col(0) = Fcol0_Transpose_D0_Block.col(0).cwiseProduct(SPK2_4_D0_Block);
    //Ptransposed_scaled_row2_D0_Block.col(0) += Fcol1_Transpose_D0_Block.col(0).cwiseProduct(SPK2_3_D0_Block);
    //Ptransposed_scaled_row2_D0_Block.col(0) += Fcol2_Transpose_D0_Block.col(0).cwiseProduct(SPK2_2_D0_Block);
    //Ptransposed_scaled_row2_D0_Block.col(1) = Fcol0_Transpose_D0_Block.col(1).cwiseProduct(SPK2_4_D0_Block);
    //Ptransposed_scaled_row2_D0_Block.col(1) += Fcol1_Transpose_D0_Block.col(1).cwiseProduct(SPK2_3_D0_Block);
    //Ptransposed_scaled_row2_D0_Block.col(1) += Fcol2_Transpose_D0_Block.col(1).cwiseProduct(SPK2_2_D0_Block);
    //Ptransposed_scaled_row2_D0_Block.col(2) = Fcol0_Transpose_D0_Block.col(2).cwiseProduct(SPK2_4_D0_Block);
    //Ptransposed_scaled_row2_D0_Block.col(2) += Fcol1_Transpose_D0_Block.col(2).cwiseProduct(SPK2_3_D0_Block);
    //Ptransposed_scaled_row2_D0_Block.col(2) += Fcol2_Transpose_D0_Block.col(2).cwiseProduct(SPK2_2_D0_Block);

    //ChVectorN<double, 16> SPK2_0_D0_Block = 0.5*D0(0)*m_GQWeight_det_J_0xi_D0.array()*(Fcol0_Transpose_D0_Block.rowwise().squaredNorm().array() - 1);
    //ChVectorN<double, 16> SPK2_1_D0_Block = 0.5*D0(1)*m_GQWeight_det_J_0xi_D0.array()*(Fcol1_Transpose_D0_Block.rowwise().squaredNorm().array() - 1);
    //ChVectorN<double, 16> SPK2_2_D0_Block = 0.5*D0(2)*m_GQWeight_det_J_0xi_D0.array()*(Fcol2_Transpose_D0_Block.rowwise().squaredNorm().array() - 1);
    //ChVectorN<double, 16> SPK2_3_D0_Block = D0(3)*m_GQWeight_det_J_0xi_D0.cwiseProduct(Fcol1_Transpose_D0_Block.cwiseProduct(Fcol2_Transpose_D0_Block).rowwise().sum());
    //ChVectorN<double, 16> SPK2_4_D0_Block = D0(4)*m_GQWeight_det_J_0xi_D0.cwiseProduct(Fcol0_Transpose_D0_Block.cwiseProduct(Fcol2_Transpose_D0_Block).rowwise().sum());
    //ChVectorN<double, 16> SPK2_5_D0_Block = D0(5)*m_GQWeight_det_J_0xi_D0.cwiseProduct(Fcol0_Transpose_D0_Block.cwiseProduct(Fcol1_Transpose_D0_Block).rowwise().sum());

    //ChMatrixNM<double, 16, 3> Ptransposed_scaled_row0_D0_Block =
    //    Fcol0_Transpose_D0_Block.array().colwise()*SPK2_0_D0_Block.array() +
    //    Fcol1_Transpose_D0_Block.array().colwise()*SPK2_5_D0_Block.array() +
    //    Fcol2_Transpose_D0_Block.array().colwise()*SPK2_4_D0_Block.array();

    //ChMatrixNM<double, 16, 3> Ptransposed_scaled_row1_D0_Block =
    //    Fcol0_Transpose_D0_Block.array().colwise()*SPK2_5_D0_Block.array() +
    //    Fcol1_Transpose_D0_Block.array().colwise()*SPK2_1_D0_Block.array() +
    //    Fcol2_Transpose_D0_Block.array().colwise()*SPK2_3_D0_Block.array();

    //ChMatrixNM<double, 16, 3> Ptransposed_scaled_row2_D0_Block =
    //    Fcol0_Transpose_D0_Block.array().colwise()*SPK2_4_D0_Block.array() +
    //    Fcol1_Transpose_D0_Block.array().colwise()*SPK2_3_D0_Block.array() +
    //    Fcol2_Transpose_D0_Block.array().colwise()*SPK2_2_D0_Block.array();

    // =============================================================================

    //ChMatrixNMc<double, 4, 3> Fcol0_Transpose_Dv_Block = m_SD_precompute_Dv_col0_block.transpose()*e_bar;
    //ChMatrixNMc<double, 4, 3> Fcol1_Transpose_Dv_Block = m_SD_precompute_Dv_col1_block.transpose()*e_bar;
    //ChMatrixNMc<double, 4, 3> Fcol2_Transpose_Dv_Block = m_SD_precompute_Dv_col2_block.transpose()*e_bar;

    //ChVectorN<double, 4> Ediag_0_Dv_Block = Fcol0_Transpose_Dv_Block.col(0).cwiseProduct(Fcol0_Transpose_Dv_Block.col(0));
    //Ediag_0_Dv_Block += Fcol0_Transpose_Dv_Block.col(1).cwiseProduct(Fcol0_Transpose_Dv_Block.col(1));
    //Ediag_0_Dv_Block += Fcol0_Transpose_Dv_Block.col(2).cwiseProduct(Fcol0_Transpose_Dv_Block.col(2));
    //Ediag_0_Dv_Block.array() -= 1;
    //Ediag_0_Dv_Block.array() *= 0.5*m_GQWeight_det_J_0xi_Dv.array();

    //ChVectorN<double, 4> Ediag_1_Dv_Block = Fcol1_Transpose_Dv_Block.col(0).cwiseProduct(Fcol1_Transpose_Dv_Block.col(0));
    //Ediag_1_Dv_Block += Fcol1_Transpose_Dv_Block.col(1).cwiseProduct(Fcol1_Transpose_Dv_Block.col(1));
    //Ediag_1_Dv_Block += Fcol1_Transpose_Dv_Block.col(2).cwiseProduct(Fcol1_Transpose_Dv_Block.col(2));
    //Ediag_1_Dv_Block.array() -= 1;
    //Ediag_1_Dv_Block.array() *= 0.5*m_GQWeight_det_J_0xi_Dv.array();

    //ChVectorN<double, 4> Ediag_2_Dv_Block = Fcol2_Transpose_Dv_Block.col(0).cwiseProduct(Fcol2_Transpose_Dv_Block.col(0));
    //Ediag_2_Dv_Block += Fcol2_Transpose_Dv_Block.col(1).cwiseProduct(Fcol2_Transpose_Dv_Block.col(1));
    //Ediag_2_Dv_Block += Fcol2_Transpose_Dv_Block.col(2).cwiseProduct(Fcol2_Transpose_Dv_Block.col(2));
    //Ediag_2_Dv_Block.array() -= 1;
    //Ediag_2_Dv_Block.array() *= 0.5*m_GQWeight_det_J_0xi_Dv.array();

    //ChVectorN<double, 4> Ediag_0_Dv_Block = 0.5 * m_GQWeight_det_J_0xi_Dv.array() * (Fcol0_Transpose_Dv_Block.rowwise().squaredNorm().array() - 1);
    //ChVectorN<double, 4> Ediag_1_Dv_Block = 0.5 * m_GQWeight_det_J_0xi_Dv.array() * (Fcol1_Transpose_Dv_Block.rowwise().squaredNorm().array() - 1);
    //ChVectorN<double, 4> Ediag_2_Dv_Block = 0.5 * m_GQWeight_det_J_0xi_Dv.array() * (Fcol2_Transpose_Dv_Block.rowwise().squaredNorm().array() - 1);

    ChVectorN<double, 4> Ediag_0_Dv_Block = Fcol0_Transpose_Dv_Block_col0.cwiseProduct(Fcol0_Transpose_Dv_Block_col0);
    Ediag_0_Dv_Block += Fcol0_Transpose_Dv_Block_col1.cwiseProduct(Fcol0_Transpose_Dv_Block_col1);
    Ediag_0_Dv_Block += Fcol0_Transpose_Dv_Block_col2.cwiseProduct(Fcol0_Transpose_Dv_Block_col2);
    Ediag_0_Dv_Block.array() -= 1;
    Ediag_0_Dv_Block *= 0.5;
    Ediag_0_Dv_Block.array() *= m_GQWeight_det_J_0xi_Dv.array();

    ChVectorN<double, 4> Ediag_1_Dv_Block = Fcol1_Transpose_Dv_Block_col0.cwiseProduct(Fcol1_Transpose_Dv_Block_col0);
    Ediag_1_Dv_Block += Fcol1_Transpose_Dv_Block_col1.cwiseProduct(Fcol1_Transpose_Dv_Block_col1);
    Ediag_1_Dv_Block += Fcol1_Transpose_Dv_Block_col2.cwiseProduct(Fcol1_Transpose_Dv_Block_col2);
    Ediag_1_Dv_Block.array() -= 1;
    Ediag_1_Dv_Block *= 0.5;
    Ediag_1_Dv_Block.array() *= m_GQWeight_det_J_0xi_Dv.array();

    ChVectorN<double, 4> Ediag_2_Dv_Block = Fcol2_Transpose_Dv_Block_col0.cwiseProduct(Fcol2_Transpose_Dv_Block_col0);
    Ediag_2_Dv_Block += Fcol2_Transpose_Dv_Block_col1.cwiseProduct(Fcol2_Transpose_Dv_Block_col1);
    Ediag_2_Dv_Block += Fcol2_Transpose_Dv_Block_col2.cwiseProduct(Fcol2_Transpose_Dv_Block_col2);
    Ediag_2_Dv_Block.array() -= 1;
    Ediag_2_Dv_Block *= 0.5;
    Ediag_2_Dv_Block.array() *= m_GQWeight_det_J_0xi_Dv.array();

    ChMatrix33<double> Dv = GetMaterial()->Get_Dv();

    m_Sdiag_0_Dv_Block = Dv(0, 0)*Ediag_0_Dv_Block + Dv(1, 0)*Ediag_1_Dv_Block + Dv(2, 0)*Ediag_2_Dv_Block;
    m_Sdiag_1_Dv_Block = Dv(0, 1)*Ediag_0_Dv_Block + Dv(1, 1)*Ediag_1_Dv_Block + Dv(2, 1)*Ediag_2_Dv_Block;
    m_Sdiag_2_Dv_Block = Dv(0, 2)*Ediag_0_Dv_Block + Dv(1, 2)*Ediag_1_Dv_Block + Dv(2, 2)*Ediag_2_Dv_Block;

    P_transpose_scaled_Block_col_ordered.block<4, 1>(48, 0) = Fcol0_Transpose_Dv_Block_col0.cwiseProduct(m_Sdiag_0_Dv_Block);
    P_transpose_scaled_Block_col_ordered.block<4, 1>(48, 1) = Fcol0_Transpose_Dv_Block_col1.cwiseProduct(m_Sdiag_0_Dv_Block);
    P_transpose_scaled_Block_col_ordered.block<4, 1>(48, 2) = Fcol0_Transpose_Dv_Block_col2.cwiseProduct(m_Sdiag_0_Dv_Block);

    P_transpose_scaled_Block_col_ordered.block<4, 1>(52, 0) = Fcol1_Transpose_Dv_Block_col0.cwiseProduct(m_Sdiag_1_Dv_Block);
    P_transpose_scaled_Block_col_ordered.block<4, 1>(52, 1) = Fcol1_Transpose_Dv_Block_col1.cwiseProduct(m_Sdiag_1_Dv_Block);
    P_transpose_scaled_Block_col_ordered.block<4, 1>(52, 2) = Fcol1_Transpose_Dv_Block_col2.cwiseProduct(m_Sdiag_1_Dv_Block);

    P_transpose_scaled_Block_col_ordered.block<4, 1>(56, 0) = Fcol2_Transpose_Dv_Block_col0.cwiseProduct(m_Sdiag_2_Dv_Block);
    P_transpose_scaled_Block_col_ordered.block<4, 1>(56, 1) = Fcol2_Transpose_Dv_Block_col1.cwiseProduct(m_Sdiag_2_Dv_Block);
    P_transpose_scaled_Block_col_ordered.block<4, 1>(56, 2) = Fcol2_Transpose_Dv_Block_col2.cwiseProduct(m_Sdiag_2_Dv_Block);

    //ChMatrixNM<double, 4, 3> Ptransposed_scaled_row0_Dv_Block;
    //Ptransposed_scaled_row0_Dv_Block.col(0) = Fcol0_Transpose_Dv_Block.col(0).cwiseProduct(Sdiag_0_Dv_Block);
    //Ptransposed_scaled_row0_Dv_Block.col(1) = Fcol0_Transpose_Dv_Block.col(1).cwiseProduct(Sdiag_0_Dv_Block);
    //Ptransposed_scaled_row0_Dv_Block.col(2) = Fcol0_Transpose_Dv_Block.col(2).cwiseProduct(Sdiag_0_Dv_Block);

    //ChMatrixNM<double, 4, 3> Ptransposed_scaled_row1_Dv_Block;
    //Ptransposed_scaled_row1_Dv_Block.col(0) = Fcol1_Transpose_Dv_Block.col(0).cwiseProduct(Sdiag_1_Dv_Block);
    //Ptransposed_scaled_row1_Dv_Block.col(1) = Fcol1_Transpose_Dv_Block.col(1).cwiseProduct(Sdiag_1_Dv_Block);
    //Ptransposed_scaled_row1_Dv_Block.col(2) = Fcol1_Transpose_Dv_Block.col(2).cwiseProduct(Sdiag_1_Dv_Block);

    //ChMatrixNM<double, 4, 3> Ptransposed_scaled_row2_Dv_Block;
    //Ptransposed_scaled_row2_Dv_Block.col(0) = Fcol2_Transpose_Dv_Block.col(0).cwiseProduct(Sdiag_2_Dv_Block);
    //Ptransposed_scaled_row2_Dv_Block.col(1) = Fcol2_Transpose_Dv_Block.col(1).cwiseProduct(Sdiag_2_Dv_Block);
    //Ptransposed_scaled_row2_Dv_Block.col(2) = Fcol2_Transpose_Dv_Block.col(2).cwiseProduct(Sdiag_2_Dv_Block);

    //ChMatrixNM<double, 4, 3> Ptransposed_scaled_row0_Dv_Block =
    //    Fcol0_Transpose_Dv_Block.array().colwise()*Sdiag_0_Dv_Block.array();
    //ChMatrixNM<double, 4, 3> Ptransposed_scaled_row1_Dv_Block =
    //    Fcol1_Transpose_Dv_Block.array().colwise()*Sdiag_1_Dv_Block.array();
    //ChMatrixNM<double, 4, 3> Ptransposed_scaled_row2_Dv_Block =
    //    Fcol2_Transpose_Dv_Block.array().colwise()*Sdiag_2_Dv_Block.array();

    // =============================================================================

    //ChMatrixNM<double, 9, 3> QiCompact =
    //    m_SD_precompute_D0_col0_block * Ptransposed_scaled_row0_D0_Block +
    //    m_SD_precompute_D0_col1_block * Ptransposed_scaled_row1_D0_Block +
    //    m_SD_precompute_D0_col2_block * Ptransposed_scaled_row2_D0_Block +
    //    m_SD_precompute_Dv_col0_block * Ptransposed_scaled_row0_Dv_Block +
    //    m_SD_precompute_Dv_col1_block * Ptransposed_scaled_row1_Dv_Block +
    //    m_SD_precompute_Dv_col2_block * Ptransposed_scaled_row2_Dv_Block;
    ChMatrixNM<double, 9, 3> QiCompact = m_SD_precompute_col_ordered * P_transpose_scaled_Block_col_ordered;
    Eigen::Map<ChVectorN<double, 27>> QiReshaped(QiCompact.data(), QiCompact.size());
    Fi = QiReshaped;
}

void ChElementBeamANCF_MT35::ComputeInternalForcesSingleGQPnt(ChMatrixNM<double, 9, 3>& QiCompact, unsigned int index, ChVectorN<double, 6>& D0, ChMatrixNMc<double, 9, 3>& e_bar, ChMatrixNMc<double, 9, 3>& e_bar_dot) {
    //ChMatrix33<double> J_0xi;               //Element Jacobian between the reference configuration and normalized configuration
    //ChMatrixNM<double, 9, 3> Sxi_D;         //Matrix of normalized shape function derivatives
    ChMatrixNMc<double, 9, 3> Sxi_D_0xi = m_SD_precompute_D0.block(0, 3*index, 9, 3);     //Matrix of normalized shape function derivatives corrected for a potentially non-straight reference configuration
    ChMatrix33<double> F;                   //Deformation Gradient tensor (non-symmetric tensor)
    ChMatrix33<double> E;                   //Green-Lagrange strain tensor (symmetric tensor) - potentially combined with some damping contributions for speed
    ChMatrixNM<double,3,3> S;                   //2nd Piola-Kirchoff stress tensor (symmetric tensor)
    ChMatrix33<double> P_transpose_scaled;  //Transpose of the 1st Piola-Kirchoff stress tensor (non-symmetric tensor) Scaled by the determinate of the element Jacobian
    ChMatrix33<double> I_3x3;               //3x3 identity matrix
    //ChMatrixNM<double, 9, 3> QiCompact;     //The compact result of the final generalized internal force computation

    ////Calculate the normalized shape function derivatives in matrix form and the corrected version of the 
    //// matrix for potentially non-straight reference configurations
    //Calc_Sxi_D(Sxi_D, xi, eta, zeta);
    //J_0xi.noalias() = m_e0_bar.transpose()*Sxi_D;
    //Sxi_D_0xi.noalias() = m_SD_precompute_D0.block(9 * index, 0, 9, 3);  //Faster to use this than to reference m_SD_precompute_D0 directly in the calculations 

    //Calculate the deformation gradient tensor and the Green-Lagrange strain tensor
    I_3x3.setIdentity();
    F.noalias() = e_bar.transpose()*Sxi_D_0xi;
    //F.noalias() = e_bar.transpose()*m_SD_precompute_D0.block(0, 3 * index, 9, 3);
    E.noalias() = 0.5*(F.transpose()*F - I_3x3);  //Faster than subtracting off identity separately term by term

                                                  //If linear Kelvin-Voigt viscoelastic material model is enabled (otherwise its pure linear elastic)
                                                  //Combine the viscous damping contribution to the Strain tensor with E (so E is no longer really "E")
    if (m_damping_enabled) {
        ChMatrix33<double> F_dot;
        F_dot.noalias() = e_bar_dot.transpose()*Sxi_D_0xi;
        //F_dot.noalias() = e_bar_dot.transpose()*m_SD_precompute_D0.block(0, 3 * index, 9, 3);

        //Combine alpha*E_dot with E (i.e E_combined = E + Alpha*E_dot)
        E += m_Alpha* 0.5* (F_dot.transpose()*F + F.transpose()*F_dot);
    }

    //2nd PK2 Stress tensor is symmetric just like the Green-Lagrange Strain tensor
    //Carry out the multiplication with the specific provided diagonal 6x6 stiffness tensor
    // (without explicitly forming the matrix multiplication in Voigt notation)
    S(0, 0) = D0(0)*E(0, 0);
    S(0, 1) = 2.0*D0(5)*E(0, 1);
    S(0, 2) = 2.0*D0(4)*E(0, 2);
    S(1, 0) = S(0, 1);
    S(1, 1) = D0(1)*E(1, 1);
    S(1, 2) = 2.0*D0(3)*E(1, 2);
    S(2, 0) = S(0, 2);
    S(2, 1) = S(1, 2);
    S(2, 2) = D0(2)*E(2, 2);

    //Calculate the transpose of the 1st Piola Kirchoff Stress Tensor
    //Scale the PK1 Stress Tensor by J_0xi.determinant() so that this multiplication is already included when calculating Qi
    // See the following reference for more details on why the PK1 Stress Tensor is
    // used for more efficient calculation of the generalized internal forces.
    //      J.Gerstmayr,A.A.Shabana,Efficient integration of the elastic forces and 
    //      thin three-dimensional beam elements in the absolute nodal coordinate formulation, 
    //      Proceedings of Multibody Dynamics 2005 ECCOMAS Thematic Conference, Madrid, Spain, 2005.
    //P_transpose_scaled.noalias() = m_GQWeight_det_J_0xi_D0(index)*S*F.transpose(); //S is symmetric, so S' = S
    //QiCompact+= Sxi_D_0xi*P_transpose_scaled;

    //P_transpose_scaled.noalias() = m_GQWeight_det_J_0xi_D0(index)*S.selfadjointView<Eigen::Upper>()*F.transpose(); //S is symmetric, so S' = S
    //QiCompact += Sxi_D_0xi*P_transpose_scaled;

    //QiCompact += m_GQWeight_det_J_0xi_D0(index)*Sxi_D_0xi*S.selfadjointView<Eigen::Upper>()*F.transpose();
    QiCompact += m_GQWeight_det_J_0xi_D0(index)*Sxi_D_0xi*S*F.transpose();
    //QiCompact += m_GQWeight_det_J_0xi_D0(index)*m_SD_precompute_D0.block(0, 3 * index, 9, 3)*S*F.transpose();
}

void ChElementBeamANCF_MT35::ComputeInternalForcesSingleGQPnt(ChMatrixNM<double, 9, 3>& QiCompact, unsigned int index, ChMatrix33<double>& Dv, ChMatrixNMc<double, 9, 3>& e_bar, ChMatrixNMc<double, 9, 3>& e_bar_dot) {
    //ChMatrix33<double> J_0xi;               //Element Jacobian between the reference configuration and normalized configuration
    //ChMatrixNM<double, 9, 3> Sxi_D;         //Matrix of normalized shape function derivatives
    ChMatrixNMc<double, 9, 3> Sxi_D_0xi = m_SD_precompute_Dv.block(0, 3*index, 9, 3);     //Matrix of normalized shape function derivatives corrected for a potentially non-straight reference configuration
    ChMatrix33<double> F;                   //Deformation Gradient tensor (non-symmetric tensor)
    ChMatrix33<double> E;                   //Green-Lagrange strain tensor (symmetric tensor) - potentially combined with some damping contributions for speed
    ChVectorN<double, 3> S;                 //2nd Piola-Kirchoff stress tensor (symmetric tensor) - Diagonal in this special case due to the structure of the 6x6 stiffness matrix
    ChMatrix33<double> P_transpose_scaled;  //Transpose of the 1st Piola-Kirchoff stress tensor (non-symmetric tensor) Scaled by the determinate of the element Jacobian
    ChMatrix33<double> I_3x3;               //3x3 identity matrix
    //ChMatrixNM<double, 9, 3> QiCompact;     //The compact result of the final generalized internal force computation

    ////Calculate the normalized shape function derivatives in matrix form and the corrected version of the 
    //// matrix for potentially non-straight reference configurations
    //Calc_Sxi_D(Sxi_D, xi, eta, zeta);
    //J_0xi.noalias() = m_e0_bar.transpose()*Sxi_D;
    //Sxi_D_0xi.noalias() = m_SD_precompute_Dv.block(9 * index, 0, 9, 3);

    //Calculate the deformation gradient tensor and the Green-Lagrange strain tensor
    I_3x3.setIdentity();
    F.noalias() = e_bar.transpose()*Sxi_D_0xi;
    //F.noalias() = e_bar.transpose()*m_SD_precompute_Dv.block(0, 3 * index, 9, 3);
    E.noalias() = 0.5*(F.transpose()*F - I_3x3);

    //If linear Kelvin-Voigt viscoelastic material model is enabled (otherwise its pure linear elastic)
    //Combine the viscous damping contribution to the Strain tensor with E (so E is no longer really "E")
    if (m_damping_enabled) {
        ChMatrix33<double> F_dot;
        //ChMatrix33<double> alpha_E_dot;
        //ChVectorN<double, 6> epsilon_dot;
        F_dot.noalias() = e_bar_dot.transpose()*Sxi_D_0xi;
        //F_dot.noalias() = e_bar_dot.transpose()*m_SD_precompute_Dv.block(0, 3 * index, 9, 3);

        //Combine alpha E_dot with E (E_combined = E + Alpha*E)
        E += m_Alpha* 0.5* (F_dot.transpose()*F + F.transpose()*F_dot);
    }

    //2nd PK2 Stress tensor is symmetric just like the Green-Lagrange Strain tensor
    //In this case its a diagonal matrix, so just compute the diagonal entries
    S.noalias() = Dv*E.diagonal();

    //Calculate the transpose of the 1st Piola Kirchoff Stress Tensor
    //Scale the PK1 Stress Tensor by J_0xi.determinant() so that this multiplication is already included when calculating Qi
    // See the following reference for more details on why the PK1 Stress Tensor is
    // used for more efficient calculation of the generalized internal forces.
    //      J.Gerstmayr,A.A.Shabana,Efficient integration of the elastic forces and 
    //      thin three-dimensional beam elements in the absolute nodal coordinate formulation, 
    //      Proceedings of Multibody Dynamics 2005 ECCOMAS Thematic Conference, Madrid, Spain, 2005.
    //P_transpose_scaled.row(0).noalias() = m_GQWeight_det_J_0xi_Dv(index)*F.col(0)*S(0);
    //P_transpose_scaled.row(1).noalias() = m_GQWeight_det_J_0xi_Dv(index)*F.col(1)*S(1);
    //P_transpose_scaled.row(2).noalias() = m_GQWeight_det_J_0xi_Dv(index)*F.col(2)*S(2);

    //P_transpose_scaled.noalias() = m_GQWeight_det_J_0xi_D0(index)*S.asDiagonal()*F.transpose(); //S is symmetric, so S' = S
    //QiCompact+= Sxi_D_0xi*P_transpose_scaled;

    QiCompact += m_GQWeight_det_J_0xi_D0(index)*Sxi_D_0xi*S.asDiagonal()*F.transpose();
    //QiCompact += m_GQWeight_det_J_0xi_D0(index)*m_SD_precompute_Dv.block(0, 3 * index, 9, 3)*S.asDiagonal()*F.transpose();

}

void ChElementBeamANCF_MT35::ComputeInternalForcesSingleGQPnt(ChMatrixNM<double, 9, 3>& QiCompact, double xi, double eta, double zeta, ChMatrixNM<double, 6, 6>& D, ChMatrixNMc<double, 9, 3>& e_bar, ChMatrixNMc<double, 9, 3>& e_bar_dot) {
    ChMatrix33<double> J_0xi;               //Element Jacobian between the reference configuration and normalized configuration
    ChMatrixNMc<double, 9, 3> Sxi_D;         //Matrix of normalized shape function derivatives
    ChMatrixNMc<double, 9, 3> Sxi_D_0xi;     //Matrix of normalized shape function derivatives corrected for a potentially non-straight reference configuration
    ChVectorN<double, 6> epsilon;           //Green-Lagrange strain tensor in Voigt notation
    ChVectorN<double, 6> sigma_PK2;         //2nd Piola-Kirchoff stress tensor in Voigt notation
    ChMatrix33<double> F;                   //Deformation Gradient tensor (non-symmetric tensor)
    ChMatrix33<double> E;                   //Green-Lagrange strain tensor (symmetric tensor) - potentially combined with some damping contributions for speed
    ChMatrix33<double> S;                   //2nd Piola-Kirchoff stress tensor (symmetric tensor)
    ChMatrix33<double> P_transpose_scaled;  //Transpose of the 1st Piola-Kirchoff stress tensor (non-symmetric tensor) Scaled by the determinate of the element Jacobian
    ChMatrix33<double> I_3x3;               //3x3 identity matrix
    //ChMatrixNM<double, 9, 3> QiCompact;     //The compact result of the final generalized internal force computation

    //Calculate the normalized shape function derivatives in matrix form and the corrected version of the 
    // matrix for potentially non-straight reference configurations
    Calc_Sxi_D(Sxi_D, xi, eta, zeta);
    J_0xi.noalias() = m_e0_bar.transpose()*Sxi_D;
    Sxi_D_0xi.noalias() = Sxi_D*J_0xi.inverse();

    //Calculate the deformation gradient tensor and the Green-Lagrange strain tensor
    I_3x3.setIdentity();
    F.noalias() = e_bar.transpose()*Sxi_D_0xi;
    E.noalias() = 0.5*(F.transpose()*F - I_3x3);
    epsilon(0) = E(0, 0);
    epsilon(1) = E(1, 1);
    epsilon(2) = E(2, 2);
    epsilon(3) = 2.0*E(1, 2);
    epsilon(4) = 2.0*E(0, 2);
    epsilon(5) = 2.0*E(0, 1);

    //Calculate the 2nd Piola-Kirchoff Stress tensor in Voigt notation
    if (m_damping_enabled) {        //If linear Kelvin-Voigt viscoelastic material model is enabled 
        ChMatrix33<double> F_dot;
        ChMatrix33<double> E_dot;
        ChVectorN<double, 6> epsilon_dot;
        F_dot.noalias() = e_bar_dot.transpose()*Sxi_D_0xi;
        E_dot.noalias() = 0.5* (F_dot.transpose()*F + F.transpose()*F_dot);
        epsilon_dot(0) = E_dot(0, 0);
        epsilon_dot(1) = E_dot(1, 1);
        epsilon_dot(2) = E_dot(2, 2);
        epsilon_dot(3) = 2.0*E_dot(1, 2);
        epsilon_dot(4) = 2.0*E_dot(0, 2);
        epsilon_dot(5) = 2.0*E_dot(0, 1);

        sigma_PK2.noalias() = D*(epsilon + m_Alpha*epsilon_dot);
    }
    else {                          //No damping, so just a linear elastic material law
        sigma_PK2.noalias() = D*epsilon;
    }

    //Now convert from Voigt notation back into matrix form
    S(0, 0) = sigma_PK2(0);
    S(1, 0) = sigma_PK2(5);
    S(2, 0) = sigma_PK2(4);
    S(0, 1) = sigma_PK2(5);
    S(1, 1) = sigma_PK2(1);
    S(2, 1) = sigma_PK2(3);
    S(0, 2) = sigma_PK2(4);
    S(1, 2) = sigma_PK2(3);
    S(2, 2) = sigma_PK2(2);

    //Calculate the transpose of the 1st Piola Kirchoff Stress Tensor
    //Scale the PK1 Stress Tensor by J_0xi.determinant() so that this multiplication is already included when calculating Qi
    // See the following reference for more details on why the PK1 Stress Tensor is
    // used for more efficient calculation of the generalized internal forces.
    //      J.Gerstmayr,A.A.Shabana,Efficient integration of the elastic forces and 
    //      thin three-dimensional beam elements in the absolute nodal coordinate formulation, 
    //      Proceedings of Multibody Dynamics 2005 ECCOMAS Thematic Conference, Madrid, Spain, 2005.
    P_transpose_scaled.noalias() = -J_0xi.determinant()*S*F.transpose(); //S is symmetric, so S' = S
    QiCompact+= Sxi_D_0xi*P_transpose_scaled;
}

// -----------------------------------------------------------------------------
// Jacobians of internal forces
// -----------------------------------------------------------------------------

void ChElementBeamANCF_MT35::ComputeInternalJacobianSingleGQPnt(ChMatrixNM<double, 27, 27>& Jac_Dense, ChMatrixNM<double, 9, 9>& Jac_Compact, double Kfactor, double Rfactor, unsigned int index, ChVectorN<double, 6>& D0, ChMatrixNMc<double, 9, 3>& e_bar, ChMatrixNMc<double, 9, 3>& e_bar_dot) {

    ChMatrixNMc<double, 9, 3> Sxi_D_0xi = m_SD_precompute_D0.block(0, 3*index, 9, 3);  //Faster to use this than to reference m_SD_precompute_D0 directly in the calculations 
    ChMatrixNM<double, 3, 3> F;                   //Deformation Gradient tensor (non-symmetric tensor)
    ChMatrixNM<double, 3, 3> E;                   //Green-Lagrange strain tensor (symmetric tensor) - potentially combined with some damping contributions for speed
    ChMatrixNM<double, 3, 3> S;                   //2nd Piola-Kirchoff stress tensor (symmetric tensor)
    ChMatrixNM<double, 3, 3> I_3x3;               //3x3 identity matrix
    ChMatrixNM<double, 6, 27> partial_epsilon_partial_e;//partial derivate of the Green-Lagrange strain tensor in Voigt notation with respect to the nodal coordinates
    ChMatrixNM<double, 6, 27> Scaled_Combined_partial_epsilon_partial_e;
    Eigen::Map<ChVectorN<double, 27>> Sxi_D_0xiReshaped(Sxi_D_0xi.data(), Sxi_D_0xi.size());
    ChMatrixNM<double, 27, 3> partial_e_tempA;//temporary matrix for calculating the partial derivatives of the strains with respect to the nodal coordinates
    Eigen::Map<ChRowVectorN<double, 81>> partial_e_tempReshapedA(partial_e_tempA.data(), partial_e_tempA.size());
    ChMatrixNM<double, 27, 3> partial_e_tempB;//temporary matrix for calculating the partial derivatives of the strains with respect to the nodal coordinates
    Eigen::Map<ChRowVectorN<double, 81>> partial_e_tempReshapedB(partial_e_tempB.data(), partial_e_tempB.size());
    ChMatrixNM<double, 27, 3> partial_e_tempC;//temporary matrix for calculating the partial derivatives of the strains with respect to the nodal coordinates
    Eigen::Map<ChRowVectorN<double, 81>> partial_e_tempReshapedC(partial_e_tempC.data(), partial_e_tempC.size());

    //Calculate the deformation gradient tensor and the Green-Lagrange strain tensor
    I_3x3.setIdentity();
    F.noalias() = e_bar.transpose()*Sxi_D_0xi;
    E.noalias() = 0.5*(F.transpose()*F - I_3x3);

    //Calculate the partial derivative of epsilon (Green-Lagrange strain tensor in Voigt notation) 
    //with respect to the nodal coordinates entry by entry in the vector epsilon
    //partial_e_tempA = Sxi_D_0xiReshaped*F.col(0).transpose();
    //partial_e_tempB = Sxi_D_0xiReshaped*F.col(1).transpose();
    //partial_e_tempC = Sxi_D_0xiReshaped*F.col(2).transpose();

    //Calculate the partial derivative of epsilon (Green-Lagrange strain tensor in Voigt notation) 
    //with respect to the nodal coordinates entry by entry in the vector epsilon
    ChRowVectorN<double, 3> Fcol0transpose = F.col(0).transpose();
    ChRowVectorN<double, 3> Fcol1transpose = F.col(1).transpose();
    ChRowVectorN<double, 3> Fcol2transpose = F.col(2).transpose();
    partial_e_tempA = Sxi_D_0xiReshaped*Fcol0transpose;
    partial_e_tempB = Sxi_D_0xiReshaped*Fcol1transpose;
    partial_e_tempC = Sxi_D_0xiReshaped*Fcol2transpose;

    partial_epsilon_partial_e.row(0) = partial_e_tempReshapedA.block(0, 0, 1, 27);
    partial_epsilon_partial_e.row(1) = partial_e_tempReshapedB.block(0, 27, 1, 27);
    partial_epsilon_partial_e.row(2) = partial_e_tempReshapedC.block(0, 54, 1, 27);
    partial_epsilon_partial_e.row(3).noalias() = partial_e_tempReshapedB.block(0, 54, 1, 27) + partial_e_tempReshapedC.block(0, 27, 1, 27);
    partial_epsilon_partial_e.row(4).noalias() = partial_e_tempReshapedA.block(0, 54, 1, 27) + partial_e_tempReshapedC.block(0, 0, 1, 27);
    partial_epsilon_partial_e.row(5).noalias() = partial_e_tempReshapedA.block(0, 27, 1, 27) + partial_e_tempReshapedB.block(0, 0, 1, 27);

    //Calculate the 2nd Piola-Kirchoff Stress tensor in Voigt notation
    if (m_damping_enabled) {        //If linear Kelvin-Voigt viscoelastic material model is enabled 
        ChMatrixNM<double, 3, 3> F_dot;
        ChMatrixNM<double, 3, 3> Scaled_Combined_F;
        Eigen::Map<ChRowVectorN<double, 9>> Scaled_Combined_FReshaped(F.data(), F.size());

        F_dot.noalias() = e_bar_dot.transpose()*Sxi_D_0xi;

        //Combine alpha*E_dot with E (i.e E_combined = E + Alpha*E_dot)
        E += m_Alpha* 0.5* (F_dot.transpose()*F + F.transpose()*F_dot);

        Scaled_Combined_F = m_GQWeight_det_J_0xi_D0(index)*((Kfactor + m_Alpha*Rfactor)*F + (m_Alpha*Kfactor)*F_dot);

        ChRowVectorN<double, 3> Scaled_Combined_Fcol0transpose = Scaled_Combined_F.col(0).transpose();
        ChRowVectorN<double, 3> Scaled_Combined_Fcol1transpose = Scaled_Combined_F.col(1).transpose();
        ChRowVectorN<double, 3> Scaled_Combined_Fcol2transpose = Scaled_Combined_F.col(2).transpose();

        //Calculate the partial derivative of epsilon (Green-Lagrange strain tensor in Voigt notation) 
        //combined with epsilon dot with the correct Jacobian weighting factors 
        //with respect to the nodal coordinates entry by entry in the vector epsilon
        //partial_e_tempA = Sxi_D_0xiReshaped*Scaled_Combined_F.col(0).transpose();
        //partial_e_tempB = Sxi_D_0xiReshaped*Scaled_Combined_F.col(1).transpose();
        //partial_e_tempC = Sxi_D_0xiReshaped*Scaled_Combined_F.col(2).transpose();
        partial_e_tempA = Sxi_D_0xiReshaped*Scaled_Combined_Fcol0transpose;
        partial_e_tempB = Sxi_D_0xiReshaped*Scaled_Combined_Fcol1transpose;
        partial_e_tempC = Sxi_D_0xiReshaped*Scaled_Combined_Fcol2transpose;

        Scaled_Combined_partial_epsilon_partial_e.row(0) = D0(0)*partial_e_tempReshapedA.block(0, 0, 1, 27);
        Scaled_Combined_partial_epsilon_partial_e.row(1) = D0(1)*partial_e_tempReshapedB.block(0, 27, 1, 27);
        Scaled_Combined_partial_epsilon_partial_e.row(2) = D0(2)*partial_e_tempReshapedC.block(0, 54, 1, 27);
        Scaled_Combined_partial_epsilon_partial_e.row(3).noalias() = D0(3)*(partial_e_tempReshapedB.block(0, 54, 1, 27) + partial_e_tempReshapedC.block(0, 27, 1, 27));
        Scaled_Combined_partial_epsilon_partial_e.row(4).noalias() = D0(4)*(partial_e_tempReshapedA.block(0, 54, 1, 27) + partial_e_tempReshapedC.block(0, 0, 1, 27));
        Scaled_Combined_partial_epsilon_partial_e.row(5).noalias() = D0(5)*(partial_e_tempReshapedA.block(0, 27, 1, 27) + partial_e_tempReshapedB.block(0, 0, 1, 27));

        //Calculate and accumulate the dense component of the Jacobian.
        Jac_Dense += partial_epsilon_partial_e.transpose()*Scaled_Combined_partial_epsilon_partial_e;

    }
    else {                          //No damping, so just a linear elastic material law
        //Scaled_Combined_partial_epsilon_partial_e.row(0) = (m_GQWeight_det_J_0xi_D0(index)*Kfactor*D0(0))*partial_epsilon_partial_e.row(0);
        //Scaled_Combined_partial_epsilon_partial_e.row(1) = (m_GQWeight_det_J_0xi_D0(index)*Kfactor*D0(1))*partial_epsilon_partial_e.row(1);
        //Scaled_Combined_partial_epsilon_partial_e.row(2) = (m_GQWeight_det_J_0xi_D0(index)*Kfactor*D0(2))*partial_epsilon_partial_e.row(2);
        //Scaled_Combined_partial_epsilon_partial_e.row(3) = (m_GQWeight_det_J_0xi_D0(index)*Kfactor*D0(3))*partial_epsilon_partial_e.row(3);
        //Scaled_Combined_partial_epsilon_partial_e.row(4) = (m_GQWeight_det_J_0xi_D0(index)*Kfactor*D0(4))*partial_epsilon_partial_e.row(4);
        //Scaled_Combined_partial_epsilon_partial_e.row(5) = (m_GQWeight_det_J_0xi_D0(index)*Kfactor*D0(5))*partial_epsilon_partial_e.row(5);

        //Calculate and accumulate the dense component of the Jacobian.
        Jac_Dense += m_GQWeight_det_J_0xi_D0(index)*Kfactor*partial_epsilon_partial_e.transpose()*D0.asDiagonal()*partial_epsilon_partial_e;

        //No timing difference between the above methods 5/3/2020
    }



    //2nd PK2 Stress tensor is symmetric just like the Green-Lagrange Strain tensor
    //Carry out the multiplication with the specific provided diagonal 6x6 stiffness tensor
    // (without explicitly forming the matrix multiplication in Voigt notation)
    //Also include the weighting factors needed for the Jacobian calculation in the next step
    S(0, 0) = (m_GQWeight_det_J_0xi_D0(index)*Kfactor)*D0(0)*E(0, 0);
    S(1, 0) = (m_GQWeight_det_J_0xi_D0(index)*Kfactor)*2.0*D0(5)*E(0, 1);
    S(2, 0) = (m_GQWeight_det_J_0xi_D0(index)*Kfactor)*2.0*D0(4)*E(0, 2);
    S(0, 1) = S(1, 0);
    S(1, 1) = (m_GQWeight_det_J_0xi_D0(index)*Kfactor)*D0(1)*E(1, 1);
    S(2, 1) = (m_GQWeight_det_J_0xi_D0(index)*Kfactor)*2.0*D0(3)*E(1, 2);
    S(0, 2) = S(2, 0);
    S(1, 2) = S(2, 1);
    S(2, 2) = (m_GQWeight_det_J_0xi_D0(index)*Kfactor)*D0(2)*E(2, 2);

    //Calculate and accumulate the compact component of the Jacobian.  This will need to 
    //be expanded to full size once all of the GQ points have been accounted for.  
    Jac_Compact += Sxi_D_0xi*S*Sxi_D_0xi.transpose();
}

void ChElementBeamANCF_MT35::ComputeInternalJacobianSingleGQPnt(ChMatrixNM<double, 27, 27>& Jac_Dense, ChMatrixNM<double, 9, 9>& Jac_Compact, double Kfactor, double Rfactor, unsigned int index, ChMatrix33<double>& Dv, ChMatrixNMc<double, 9, 3>& e_bar, ChMatrixNMc<double, 9, 3>& e_bar_dot) {

    ChMatrixNMc<double, 9, 3> Sxi_D_0xi = m_SD_precompute_Dv.block(0, 3*index, 9, 3);     //Matrix of normalized shape function derivatives corrected for a potentially non-straight reference configuration
                                            //Faster to use this than to reference m_SD_precompute_D0 directly in the calculations 
    ChMatrixNM<double, 3, 3> F;                   //Deformation Gradient tensor (non-symmetric tensor)
    ChMatrix33<double> E;                   //Green-Lagrange strain tensor (symmetric tensor) - potentially combined with some damping contributions for speed
    ChVectorN<double, 3> S;                 //2nd Piola-Kirchoff stress tensor (symmetric tensor) - Diagonal in this special case due to the structure of the 6x6 stiffness matrix
    ChMatrix33<double> I_3x3;               //3x3 identity matrix
    ChMatrixNM<double, 3, 27> partial_epsilon_partial_e;//partial derivate of the Green-Lagrange strain tensor in Voigt notation with respect to the nodal coordinates
    ChMatrixNM<double, 9, 3> partial_e_temp0;//temporary matrix for calculating the partial derivatives of the strains with respect to the nodal coordinates
    Eigen::Map<ChRowVectorN<double, 27>> partial_e_tempReshaped0(partial_e_temp0.data(), partial_e_temp0.size());
    ChMatrixNM<double, 9, 3> partial_e_temp1;//temporary matrix for calculating the partial derivatives of the strains with respect to the nodal coordinates
    Eigen::Map<ChRowVectorN<double, 27>> partial_e_tempReshaped1(partial_e_temp1.data(), partial_e_temp1.size());
    ChMatrixNM<double, 9, 3> partial_e_temp2;//temporary matrix for calculating the partial derivatives of the strains with respect to the nodal coordinates
    Eigen::Map<ChRowVectorN<double, 27>> partial_e_tempReshaped2(partial_e_temp2.data(), partial_e_temp2.size());


    //Calculate the deformation gradient tensor and the Green-Lagrange strain tensor
    I_3x3.setIdentity();
    F.noalias() = e_bar.transpose()*Sxi_D_0xi;
    E.noalias() = 0.5*(F.transpose()*F - I_3x3);

    //Calculate the partial derivative of epsilon (Green-Lagrange strain tensor in Voigt notation) 
    //with respect to the nodal coordinates entry by entry in the vector epsilon
    //Only need the first 3 terms in this special case due to the structure of the 6x6 stiffness matrix
    //partial_e_temp0 = Sxi_D_0xi.col(0)*F.col(0).transpose();
    //partial_epsilon_partial_e.row(0) = partial_e_tempReshaped0;
    //partial_e_temp1 = Sxi_D_0xi.col(1)*F.col(1).transpose();
    //partial_epsilon_partial_e.row(1) = partial_e_tempReshaped1;
    //partial_e_temp2 = Sxi_D_0xi.col(2)*F.col(2).transpose();
    //partial_epsilon_partial_e.row(2) = partial_e_tempReshaped2;

    //Calculate the partial derivative of epsilon (Green-Lagrange strain tensor in Voigt notation) 
    //with respect to the nodal coordinates entry by entry in the vector epsilon
    //Only need the first 3 terms in this special case due to the structure of the 6x6 stiffness matrix
    ChRowVectorN<double, 3> Fcol0transpose = F.col(0).transpose();
    ChRowVectorN<double, 3> Fcol1transpose = F.col(1).transpose();
    ChRowVectorN<double, 3> Fcol2transpose = F.col(2).transpose();
    partial_e_temp0 = Sxi_D_0xi.col(0)*Fcol0transpose;
    partial_epsilon_partial_e.row(0) = partial_e_tempReshaped0;
    partial_e_temp1 = Sxi_D_0xi.col(1)*Fcol1transpose;
    partial_epsilon_partial_e.row(1) = partial_e_tempReshaped1;
    partial_e_temp2 = Sxi_D_0xi.col(2)*Fcol2transpose;
    partial_epsilon_partial_e.row(2) = partial_e_tempReshaped2;


    //Calculate the 2nd Piola-Kirchoff Stress tensor in Voigt notation
    if (m_damping_enabled) {        //If linear Kelvin-Voigt viscoelastic material model is enabled 
        ChMatrix33<double> F_dot;
        ChMatrix33<double> E_dot;
        ChMatrix33<double> Scaled_Combined_F;
        ChMatrixNM<double, 3, 27> Scaled_Combined_partial_epsilon_partial_e;

        F_dot.noalias() = e_bar_dot.transpose()*Sxi_D_0xi;
        E_dot.noalias() = 0.5* (F_dot.transpose()*F + F.transpose()*F_dot);

        S.noalias() = Dv*(E.diagonal() + m_Alpha*E_dot.diagonal());

        Scaled_Combined_F = m_GQWeight_det_J_0xi_Dv(index)*((Kfactor + m_Alpha*Rfactor)*F + (m_Alpha*Kfactor)*F_dot);

        ChRowVectorN<double, 3> Scaled_Combined_Fcol0transpose = Scaled_Combined_F.col(0).transpose();
        ChRowVectorN<double, 3> Scaled_Combined_Fcol1transpose = Scaled_Combined_F.col(1).transpose();
        ChRowVectorN<double, 3> Scaled_Combined_Fcol2transpose = Scaled_Combined_F.col(2).transpose();

        //Calculate the partial derivative of epsilon (Green-Lagrange strain tensor in Voigt notation) 
        //combined with epsilon dot with the correct Jacobian weighting factors 
        //with respect to the nodal coordinates entry by entry in the vector epsilon
        //partial_e_temp0 = Sxi_D_0xi.col(0)*Scaled_Combined_F.col(0).transpose();
        //Scaled_Combined_partial_epsilon_partial_e.row(0) = partial_e_tempReshaped0;
        //partial_e_temp1 = Sxi_D_0xi.col(1)*Scaled_Combined_F.col(1).transpose();
        //Scaled_Combined_partial_epsilon_partial_e.row(1) = partial_e_tempReshaped1;
        //partial_e_temp2 = Sxi_D_0xi.col(2)*Scaled_Combined_F.col(2).transpose();
        //Scaled_Combined_partial_epsilon_partial_e.row(2) = partial_e_tempReshaped2;
        partial_e_temp0 = Sxi_D_0xi.col(0)*Scaled_Combined_Fcol0transpose;
        Scaled_Combined_partial_epsilon_partial_e.row(0) = partial_e_tempReshaped0;
        partial_e_temp1 = Sxi_D_0xi.col(1)*Scaled_Combined_Fcol1transpose;
        Scaled_Combined_partial_epsilon_partial_e.row(1) = partial_e_tempReshaped1;
        partial_e_temp2 = Sxi_D_0xi.col(2)*Scaled_Combined_Fcol2transpose;
        Scaled_Combined_partial_epsilon_partial_e.row(2) = partial_e_tempReshaped2;

        //Calculate and accumulate the dense component of the Jacobian.
        Jac_Dense += partial_epsilon_partial_e.transpose()*Dv*Scaled_Combined_partial_epsilon_partial_e;

    }
    else {                          //No damping, so just a linear elastic material law
        S.noalias() = Dv*E.diagonal();

        //Calculate and accumulate the dense component of the Jacobian.
        Jac_Dense += partial_epsilon_partial_e.transpose()*(m_GQWeight_det_J_0xi_Dv(index)*Kfactor*Dv)*partial_epsilon_partial_e;
    }

    //Calculate and accumulate the compact component of the Jacobian.  This will need to 
    //be expanded to full size once all of the GQ points have been accounted for. 
    //Jac_Compact += Sxi_D_0xi*(m_GQWeight_det_J_0xi_Dv(index)*Kfactor*S.asDiagonal())*Sxi_D_0xi.transpose();
    //Jac_Compact += (m_GQWeight_det_J_0xi_Dv(index)*Kfactor)*(S(0)*Sxi_D_0xi.col(0)*Sxi_D_0xi.col(0).transpose() + 
    //    S(1)*Sxi_D_0xi.col(1)*Sxi_D_0xi.col(1).transpose() + S(2)*Sxi_D_0xi.col(2)*Sxi_D_0xi.col(2).transpose());
    Jac_Compact += m_GQWeight_det_J_0xi_Dv(index)*Kfactor*Sxi_D_0xi*S.asDiagonal()*Sxi_D_0xi.transpose();
}

void ChElementBeamANCF_MT35::ComputeInternalJacobianSingleGQPnt(ChMatrixNM<double, 27, 27>& Jac_Dense, ChMatrixNM<double, 9, 9>& Jac_Compact, double Kfactor, double Rfactor, double GQ_weight, double xi, double eta, double zeta, ChMatrixNM<double, 6, 6>& D, ChMatrixNMc<double, 9, 3>& e_bar, ChMatrixNMc<double, 9, 3>& e_bar_dot){
    ChMatrix33<double> J_0xi;               //Element Jacobian between the reference configuration and normalized configuration
    ChMatrixNMc<double, 9, 3> Sxi_D;         //Matrix of normalized shape function derivatives
    ChMatrixNMc<double, 9, 3> Sxi_D_0xi;     //Matrix of normalized shape function derivatives corrected for a potentially non-straight reference configuration
    ChVectorN<double, 6> epsilon;           //Green-Lagrange strain tensor in Voigt notation
    ChVectorN<double, 6> sigma_PK2;         //2nd Piola-Kirchoff stress tensor in Voigt notation
    ChMatrix33<double> F;                   //Deformation Gradient tensor (non-symmetric tensor)
    ChMatrix33<double> E;                   //Green-Lagrange strain tensor (symmetric tensor) - potentially combined with some damping contributions for speed
    ChMatrix33<double> S;                   //2nd Piola-Kirchoff stress tensor (symmetric tensor)
    ChMatrix33<double> I_3x3;               //3x3 identity matrix
    ChMatrixNM<double, 6, 27> partial_epsilon_partial_e;//partial derivate of the Green-Lagrange strain tensor in Voigt notation with respect to the nodal coordinates
    Eigen::Map<ChVectorN<double, 27>> Sxi_D_0xiReshaped(Sxi_D_0xi.data(), Sxi_D_0xi.size());
    ChMatrixNM<double, 27, 3> partial_e_tempA;//temporary matrix for calculating the partial derivatives of the strains with respect to the nodal coordinates
    Eigen::Map<ChRowVectorN<double, 81>> partial_e_tempReshapedA(partial_e_tempA.data(), partial_e_tempA.size());
    ChMatrixNM<double, 27, 3> partial_e_tempB;//temporary matrix for calculating the partial derivatives of the strains with respect to the nodal coordinates
    Eigen::Map<ChRowVectorN<double, 81>> partial_e_tempReshapedB(partial_e_tempB.data(), partial_e_tempB.size());
    ChMatrixNM<double, 27, 3> partial_e_tempC;//temporary matrix for calculating the partial derivatives of the strains with respect to the nodal coordinates
    Eigen::Map<ChRowVectorN<double, 81>> partial_e_tempReshapedC(partial_e_tempC.data(), partial_e_tempC.size());


    //Calculate the normalized shape function derivatives in matrix form and the corrected version of the 
    // matrix for potentially non-straight reference configurations
    Calc_Sxi_D(Sxi_D, xi, eta, zeta);
    J_0xi.noalias() = m_e0_bar.transpose()*Sxi_D;
    Sxi_D_0xi.noalias() = Sxi_D*J_0xi.inverse();
    double GQWeight_det_J_0xi = GQ_weight*J_0xi.determinant();

    //Calculate the deformation gradient tensor and the Green-Lagrange strain tensor
    I_3x3.setIdentity();
    F.noalias() = e_bar.transpose()*Sxi_D_0xi;
    E.noalias() = 0.5*(F.transpose()*F - I_3x3);
    epsilon(0) = E(0, 0);
    epsilon(1) = E(1, 1);
    epsilon(2) = E(2, 2);
    epsilon(3) = 2.0*E(1, 2);
    epsilon(4) = 2.0*E(0, 2);
    epsilon(5) = 2.0*E(0, 1);

    //Calculate the partial derivative of epsilon (Green-Lagrange strain tensor in Voigt notation) 
    //with respect to the nodal coordinates entry by entry in the vector epsilon
    partial_e_tempA = Sxi_D_0xiReshaped*F.col(0).transpose();
    partial_e_tempB = Sxi_D_0xiReshaped*F.col(1).transpose();
    partial_e_tempC = Sxi_D_0xiReshaped*F.col(2).transpose();

    partial_epsilon_partial_e.row(0) = partial_e_tempReshapedA.block(0, 0, 1, 27);
    partial_epsilon_partial_e.row(1) = partial_e_tempReshapedB.block(0, 27, 1, 27);
    partial_epsilon_partial_e.row(2) = partial_e_tempReshapedC.block(0, 54, 1, 27);
    partial_epsilon_partial_e.row(3).noalias() = partial_e_tempReshapedB.block(0, 54, 1, 27) + partial_e_tempReshapedC.block(0, 27, 1, 27);
    partial_epsilon_partial_e.row(4).noalias() = partial_e_tempReshapedA.block(0, 54, 1, 27) + partial_e_tempReshapedC.block(0, 0, 1, 27);
    partial_epsilon_partial_e.row(5).noalias() = partial_e_tempReshapedA.block(0, 27, 1, 27) + partial_e_tempReshapedB.block(0, 0, 1, 27);

    //Calculate the 2nd Piola-Kirchoff Stress tensor in Voigt notation
    if (m_damping_enabled) {        //If linear Kelvin-Voigt viscoelastic material model is enabled 
        ChMatrix33<double> F_dot;
        ChMatrix33<double> E_dot;
        ChVectorN<double, 6> epsilon_dot;
        ChMatrix33<double> Scaled_Combined_F;
        ChMatrixNM<double, 6, 27> Scaled_Combined_partial_epsilon_partial_e;

        F_dot.noalias() = e_bar_dot.transpose()*Sxi_D_0xi;
        E_dot.noalias() = 0.5* (F_dot.transpose()*F + F.transpose()*F_dot);
        epsilon_dot(0) = E_dot(0, 0);
        epsilon_dot(1) = E_dot(1, 1);
        epsilon_dot(2) = E_dot(2, 2);
        epsilon_dot(3) = 2.0*E_dot(1, 2);
        epsilon_dot(4) = 2.0*E_dot(0, 2);
        epsilon_dot(5) = 2.0*E_dot(0, 1);

        sigma_PK2.noalias() = D*(epsilon + m_Alpha*epsilon_dot);

        Scaled_Combined_F = GQWeight_det_J_0xi*((Kfactor + m_Alpha*Rfactor)*F + (m_Alpha*Kfactor)*F_dot);

        //Calculate the partial derivative of epsilon (Green-Lagrange strain tensor in Voigt notation) 
        //combined with epsilon dot with the correct Jacobian weighting factors 
        //with respect to the nodal coordinates entry by entry in the vector epsilon
        partial_e_tempA = Sxi_D_0xiReshaped*Scaled_Combined_F.col(0).transpose();
        partial_e_tempB = Sxi_D_0xiReshaped*Scaled_Combined_F.col(1).transpose();
        partial_e_tempC = Sxi_D_0xiReshaped*Scaled_Combined_F.col(2).transpose();

        Scaled_Combined_partial_epsilon_partial_e.row(0) = partial_e_tempReshapedA.block(0, 0, 1, 27);
        Scaled_Combined_partial_epsilon_partial_e.row(1) = partial_e_tempReshapedB.block(0, 27, 1, 27);
        Scaled_Combined_partial_epsilon_partial_e.row(2) = partial_e_tempReshapedC.block(0, 54, 1, 27);
        Scaled_Combined_partial_epsilon_partial_e.row(3).noalias() = partial_e_tempReshapedB.block(0, 54, 1, 27) + partial_e_tempReshapedC.block(0, 27, 1, 27);
        Scaled_Combined_partial_epsilon_partial_e.row(4).noalias() = partial_e_tempReshapedA.block(0, 54, 1, 27) + partial_e_tempReshapedC.block(0, 0, 1, 27);
        Scaled_Combined_partial_epsilon_partial_e.row(5).noalias() = partial_e_tempReshapedA.block(0, 27, 1, 27) + partial_e_tempReshapedB.block(0, 0, 1, 27);

        //Calculate and accumulate the dense component of the Jacobian.
        Jac_Dense += partial_epsilon_partial_e.transpose()*D*Scaled_Combined_partial_epsilon_partial_e;

    }
    else {                          //No damping, so just a linear elastic material law
        sigma_PK2.noalias() = D*epsilon;

        //Calculate and accumulate the dense component of the Jacobian.
        Jac_Dense += partial_epsilon_partial_e.transpose()*(GQWeight_det_J_0xi*Kfactor*D)*partial_epsilon_partial_e;
    }

    //Now convert from Voigt notation back into matrix form
    S(0, 0) = sigma_PK2(0);
    S(1, 0) = sigma_PK2(5);
    S(2, 0) = sigma_PK2(4);
    S(0, 1) = sigma_PK2(5);
    S(1, 1) = sigma_PK2(1);
    S(2, 1) = sigma_PK2(3);
    S(0, 2) = sigma_PK2(4);
    S(1, 2) = sigma_PK2(3);
    S(2, 2) = sigma_PK2(2);

    //Calculate and accumulate the compact component of the Jacobian.  This will need to 
    //be expanded to full size once all of the GQ points have been accounted for.  
    Jac_Compact += (GQWeight_det_J_0xi*Kfactor)*Sxi_D_0xi*S*Sxi_D_0xi.transpose();

}

void ChElementBeamANCF_MT35::ComputeInternalJacobians(ChMatrixNM<double, 27, 27>& JacobianMatrix, double Kfactor, double Rfactor) {
    // The integrated quantity represents the 27x27 Jacobian
    //      Kfactor * [K] + Rfactor * [R]


    ChVectorDynamic<double> FiOrignal(27);
    ChVectorDynamic<double> FiDelta(27);
    //ChMatrixNMc<double, 9, 3> e_bar;
    //ChMatrixNMc<double, 9, 3> e_bar_dot;

    //CalcCoordMatrix(e_bar);
    //CalcCoordDerivMatrix(e_bar_dot);

    double delta = 1e-6;

    //Compute the Jacobian via numerical differentiation of the generalized internal force vector
    //Since the generalized force vector due to gravity is a constant, it doesn't affect this 
    //Jacobian calculation
    //Runs faster if the internal force with or without damping calculations are not combined into the same function using the common calculations with an if statement for the damping in the middle to calculate the different P_transpose_scaled_Block components
    if (m_damping_enabled) {
        ChMatrixNMc<double, 9, 6> ebar_ebardot;
        CalcCombinedCoordMatrix(ebar_ebardot);

        ComputeInternalForcesAtState(FiOrignal, ebar_ebardot);
        for (unsigned int i = 0; i < 27; i++) {
            ebar_ebardot(i / 3, i % 3) = ebar_ebardot(i / 3, i % 3) + delta;
            ComputeInternalForcesAtState(FiDelta, ebar_ebardot);
            JacobianMatrix.col(i) = -Kfactor / delta * (FiDelta - FiOrignal);
            ebar_ebardot(i / 3, i % 3) = ebar_ebardot(i / 3, i % 3) - delta;

            ebar_ebardot(i / 3, (i % 3) + 3) = ebar_ebardot(i / 3, (i % 3) + 3) + delta;
            ComputeInternalForcesAtState(FiDelta, ebar_ebardot);
            JacobianMatrix.col(i) += -Rfactor / delta * (FiDelta - FiOrignal);
            ebar_ebardot(i / 3, (i % 3) + 3) = ebar_ebardot(i / 3, (i % 3) + 3) - delta;
        }
    }
    else {
        ChMatrixNMc<double, 9, 3> e_bar;
        CalcCoordMatrix(e_bar);

        ComputeInternalForcesAtStateNoDamping(FiOrignal, e_bar);
        for (unsigned int i = 0; i < 27; i++) {
            e_bar(i / 3, i % 3) = e_bar(i / 3, i % 3) + delta;
            ComputeInternalForcesAtStateNoDamping(FiDelta, e_bar);
            JacobianMatrix.col(i) = -Kfactor / delta * (FiDelta - FiOrignal);
            e_bar(i / 3, i % 3) = e_bar(i / 3, i % 3) - delta;
        }
    }
}

void ChElementBeamANCF_MT35::ComputeInternalJacobiansAnalytic(ChMatrixNM<double, 27, 27>& JacobianMatrix, double Kfactor, double Rfactor) {
    ChMatrixNMc<double, 9, 3> e_bar;
    ChMatrixNMc<double, 9, 3> e_bar_dot;

    CalcCoordMatrix(e_bar);
    CalcCoordDerivMatrix(e_bar_dot);

    //Set JacobianMatrix to zero since the results from each GQ point will be added to this matrix
    JacobianMatrix.setZero();
    ChMatrixNM<double, 9, 9> JacobianMatrixCompact;
    JacobianMatrixCompact.setZero();

    ChVectorN<double, 6> D0 = GetMaterial()->Get_D0();
    ChMatrix33<double> Dv = GetMaterial()->Get_Dv();

    //Calculate the portion of the Selective Reduced Integration that does account for the Poisson effect
    for (unsigned int index = 0; index < m_GQWeight_det_J_0xi_D0.size(); index++) {
        ComputeInternalJacobianSingleGQPnt(JacobianMatrix, JacobianMatrixCompact, -Kfactor, -Rfactor, index, D0, e_bar, e_bar_dot);
    }

    //Calculate the portion of the Selective Reduced Integration that account for the Poisson effect, but only on the beam axis
    if (GetStrainFormulation() == ChElementBeamANCF_MT35::StrainFormulation::CMPoisson) {
        for (unsigned int index = 0; index < m_GQWeight_det_J_0xi_Dv.size(); index++) {
            ComputeInternalJacobianSingleGQPnt(JacobianMatrix, JacobianMatrixCompact, -Kfactor, -Rfactor, index, Dv, e_bar, e_bar_dot);
        }
    }

    //Inflate the Mass Matrix since it is stored in compact form.
    //In MATLAB notation:	 
    //JacobianMatrix(1:3:end,1:3:end) = JacobianMatrix(1:3:end,1:3:end) + JacobianMatrixCompact;
    //JacobianMatrix(2:3:end,2:3:end) = JacobianMatrix(2:3:end,2:3:end) + JacobianMatrixCompact;
    //JacobianMatrix(3:3:end,3:3:end) = JacobianMatrix(3:3:end,3:3:end) + JacobianMatrixCompact;
    for (unsigned int i = 0; i < 9; i++) {
        for (unsigned int j = 0; j < 9; j++) {
            JacobianMatrix(3 * i, 3 * j) += JacobianMatrixCompact(i, j);
            JacobianMatrix(3 * i + 1, 3 * j + 1) += JacobianMatrixCompact(i, j);
            JacobianMatrix(3 * i + 2, 3 * j + 2) += JacobianMatrixCompact(i, j);
        }
    }



}

// -----------------------------------------------------------------------------
// Shape functions
// -----------------------------------------------------------------------------

// 3x27 Sparse Form of the Normalized Shape Functions
// [s1*I_3x3, s2*I_3x3, s3*I_3x3, ...]
void ChElementBeamANCF_MT35::Calc_Sxi(ChMatrixNM<double, 3, 27>& Sxi, double xi, double eta, double zeta) {
    ChVectorN<double, 9> Sxi_compact;
    Calc_Sxi_compact(Sxi_compact, xi, eta, zeta);
    Sxi.setZero();

    //// MIKE Clean-up when slicing becomes available in Eigen 3.4
    for (unsigned int s = 0; s < Sxi_compact.size(); s++) {
        Sxi(0, 0 + (3 * s)) = Sxi_compact(s);
        Sxi(1, 1 + (3 * s)) = Sxi_compact(s);
        Sxi(2, 2 + (3 * s)) = Sxi_compact(s);
    }
}

// 9x1 Vector Form of the Normalized Shape Functions
// [s1; s2; s3; ...]
void ChElementBeamANCF_MT35::Calc_Sxi_compact(ChVectorN<double, 9>& Sxi_compact, double xi, double eta, double zeta) {
    Sxi_compact(0) = 0.5*(xi*xi - xi);
    Sxi_compact(1) = 0.25*m_thicknessY*eta*(xi*xi - xi);
    Sxi_compact(2) = 0.25*m_thicknessZ*zeta*(xi*xi - xi);
    Sxi_compact(3) = 0.5*(xi*xi + xi);
    Sxi_compact(4) = 0.25*m_thicknessY*eta*(xi*xi + xi);
    Sxi_compact(5) = 0.25*m_thicknessZ*zeta*(xi*xi + xi);
    Sxi_compact(6) = 1.0 - xi*xi;
    Sxi_compact(7) = 0.5*m_thicknessY*eta*(1.0 - xi*xi);
    Sxi_compact(8) = 0.5*m_thicknessZ*zeta*(1.0 - xi*xi);
}

//Calculate the 27x3 Compact Shape Function Derivative Matrix modified by the inverse of the element Jacobian
//            [partial(s_1)/(partial xi)      partial(s_2)/(partial xi)     ...]T
//Sxi_D_0xi = [partial(s_1)/(partial eta)     partial(s_2)/(partial eta)    ...]  J_0xi^(-1)
//            [partial(s_1)/(partial zeta)    partial(s_2)/(partial zeta)   ...]
// See: J.Gerstmayr,A.A.Shabana,Efficient integration of the elastic forces and 
//      thin three-dimensional beam elements in the absolute nodal coordinate formulation, 
//      Proceedings of Multibody Dynamics 2005 ECCOMAS Thematic Conference, Madrid, Spain, 2005.

void ChElementBeamANCF_MT35::Calc_Sxi_D(ChMatrixNMc<double, 9, 3>& Sxi_D, double xi, double eta, double zeta) {
    Sxi_D(0, 0) = xi - 0.5;
    Sxi_D(1, 0) = 0.25*m_thicknessY*eta*(2.0*xi - 1.0);
    Sxi_D(2, 0) = 0.25*m_thicknessZ*zeta*(2.0*xi - 1.0);
    Sxi_D(3, 0) = xi + 0.5;
    Sxi_D(4, 0) = 0.25*m_thicknessY*eta*(2.0*xi + 1.0);
    Sxi_D(5, 0) = 0.25*m_thicknessZ*zeta*(2.0*xi + 1.0);
    Sxi_D(6, 0) = -2.0*xi;
    Sxi_D(7, 0) = -m_thicknessY*eta*xi;
    Sxi_D(8, 0) = -m_thicknessZ*zeta*xi;

    Sxi_D(0, 1) = 0.0;
    Sxi_D(1, 1) = 0.25*m_thicknessY*(xi*xi - xi);
    Sxi_D(2, 1) = 0.0;
    Sxi_D(3, 1) = 0.0;
    Sxi_D(4, 1) = 0.25*m_thicknessY*(xi*xi + xi);
    Sxi_D(5, 1) = 0.0;
    Sxi_D(6, 1) = 0.0;
    Sxi_D(7, 1) = 0.5*m_thicknessY*(1 - xi*xi);
    Sxi_D(8, 1) = 0.0;

    Sxi_D(0, 2) = 0.0;
    Sxi_D(1, 2) = 0.0;
    Sxi_D(2, 2) = 0.25*m_thicknessZ*(xi*xi - xi);
    Sxi_D(3, 2) = 0.0;
    Sxi_D(4, 2) = 0.0;
    Sxi_D(5, 2) = 0.25*m_thicknessZ*(xi*xi + xi);
    Sxi_D(6, 2) = 0.0;
    Sxi_D(7, 2) = 0.0;
    Sxi_D(8, 2) = 0.5*m_thicknessZ*(1 - xi*xi);
}


// -----------------------------------------------------------------------------
// Helper functions
// -----------------------------------------------------------------------------

void ChElementBeamANCF_MT35::CalcCoordMatrix(ChMatrixNMc<double, 9, 3>& e) {
    e.row(0) = m_nodes[0]->GetPos().eigen();
    e.row(1) = m_nodes[0]->GetD().eigen();
    e.row(2) = m_nodes[0]->GetDD().eigen();

    e.row(3) = m_nodes[1]->GetPos().eigen();
    e.row(4) = m_nodes[1]->GetD().eigen();
    e.row(5) = m_nodes[1]->GetDD().eigen();

    e.row(6) = m_nodes[2]->GetPos().eigen();
    e.row(7) = m_nodes[2]->GetD().eigen();
    e.row(8) = m_nodes[2]->GetDD().eigen();
}

void ChElementBeamANCF_MT35::CalcCoordVector(ChVectorN<double, 27>& e) {
    e.segment(0, 3) = m_nodes[0]->GetPos().eigen();
    e.segment(3, 3) = m_nodes[0]->GetD().eigen();
    e.segment(6, 3) = m_nodes[0]->GetDD().eigen();

    e.segment(9, 3) = m_nodes[1]->GetPos().eigen();
    e.segment(12, 3) = m_nodes[1]->GetD().eigen();
    e.segment(15, 3) = m_nodes[1]->GetDD().eigen();

    e.segment(18, 3) = m_nodes[2]->GetPos().eigen();
    e.segment(21, 3) = m_nodes[2]->GetD().eigen();
    e.segment(24, 3) = m_nodes[2]->GetDD().eigen();
}

void ChElementBeamANCF_MT35::CalcCoordDerivMatrix(ChMatrixNMc<double, 9, 3>& edot) {
    edot.row(0) = m_nodes[0]->GetPos_dt().eigen();
    edot.row(1) = m_nodes[0]->GetD_dt().eigen();
    edot.row(2) = m_nodes[0]->GetDD_dt().eigen();

    edot.row(3) = m_nodes[1]->GetPos_dt().eigen();
    edot.row(4) = m_nodes[1]->GetD_dt().eigen();
    edot.row(5) = m_nodes[1]->GetDD_dt().eigen();

    edot.row(6) = m_nodes[2]->GetPos_dt().eigen();
    edot.row(7) = m_nodes[2]->GetD_dt().eigen();
    edot.row(8) = m_nodes[2]->GetDD_dt().eigen();
}

void ChElementBeamANCF_MT35::CalcCoordDerivVector(ChVectorN<double, 27>& edot) {
    edot.segment(0, 3) = m_nodes[0]->GetPos_dt().eigen();
    edot.segment(3, 3) = m_nodes[0]->GetD_dt().eigen();
    edot.segment(6, 3) = m_nodes[0]->GetDD_dt().eigen();

    edot.segment(9, 3) = m_nodes[1]->GetPos_dt().eigen();
    edot.segment(12, 3) = m_nodes[1]->GetD_dt().eigen();
    edot.segment(15, 3) = m_nodes[1]->GetDD_dt().eigen();

    edot.segment(18, 3) = m_nodes[2]->GetPos_dt().eigen();
    edot.segment(21, 3) = m_nodes[2]->GetD_dt().eigen();
    edot.segment(24, 3) = m_nodes[2]->GetDD_dt().eigen();
}

void ChElementBeamANCF_MT35::CalcCombinedCoordMatrix(ChMatrixNMc<double, 9, 6>& ebar_ebardot) {
    ebar_ebardot.block<1, 3>(0, 0) = m_nodes[0]->GetPos().eigen();
    ebar_ebardot.block<1, 3>(0, 3) = m_nodes[0]->GetPos_dt().eigen();
    ebar_ebardot.block<1, 3>(1, 0) = m_nodes[0]->GetD().eigen();
    ebar_ebardot.block<1, 3>(1, 3) = m_nodes[0]->GetD_dt().eigen();
    ebar_ebardot.block<1, 3>(2, 0) = m_nodes[0]->GetDD().eigen();
    ebar_ebardot.block<1, 3>(2, 3) = m_nodes[0]->GetDD_dt().eigen();

    ebar_ebardot.block<1, 3>(3, 0) = m_nodes[1]->GetPos().eigen();
    ebar_ebardot.block<1, 3>(3, 3) = m_nodes[1]->GetPos_dt().eigen();
    ebar_ebardot.block<1, 3>(4, 0) = m_nodes[1]->GetD().eigen();
    ebar_ebardot.block<1, 3>(4, 3) = m_nodes[1]->GetD_dt().eigen();
    ebar_ebardot.block<1, 3>(5, 0) = m_nodes[1]->GetDD().eigen();
    ebar_ebardot.block<1, 3>(5, 3) = m_nodes[1]->GetDD_dt().eigen();

    ebar_ebardot.block<1, 3>(6, 0) = m_nodes[2]->GetPos().eigen();
    ebar_ebardot.block<1, 3>(6, 3) = m_nodes[2]->GetPos_dt().eigen();
    ebar_ebardot.block<1, 3>(7, 0) = m_nodes[2]->GetD().eigen();
    ebar_ebardot.block<1, 3>(7, 3) = m_nodes[2]->GetD_dt().eigen();
    ebar_ebardot.block<1, 3>(8, 0) = m_nodes[2]->GetDD().eigen();
    ebar_ebardot.block<1, 3>(8, 3) = m_nodes[2]->GetDD_dt().eigen();
}



//Calculate the 3x3 Element Jacobian at the given point (xi,eta,zeta) in the element
void ChElementBeamANCF_MT35::Calc_J_0xi(ChMatrix33<double>& J_0xi, double xi, double eta, double zeta) {
    ChMatrixNMc<double, 9, 3> Sxi_D;
    Calc_Sxi_D(Sxi_D, xi, eta, zeta);

    J_0xi = m_e0_bar.transpose()*Sxi_D;
}

//Calculate the determinate of the 3x3 Element Jacobian at the given point (xi,eta,zeta) in the element
double ChElementBeamANCF_MT35::Calc_det_J_0xi(double xi, double eta, double zeta) {
    ChMatrixNMc<double, 9, 3> Sxi_D;
    ChMatrix33<double> J_0xi;

    Calc_Sxi_D(Sxi_D, xi, eta, zeta);
    J_0xi = m_e0_bar.transpose()*Sxi_D;
    return(J_0xi.determinant());
}

// -----------------------------------------------------------------------------
// Interface to ChElementShell base class
// -----------------------------------------------------------------------------
//ChVector<> ChElementBeamANCF_MT35::EvaluateBeamSectionStrains() {
//    // Element shape function
//    ShapeVector N;
//    this->ShapeFunctions(N, 0, 0, 0);
//
//    // Determinant of position vector gradient matrix: Initial configuration
//    ShapeVector Nx;
//    ShapeVector Ny;
//    ShapeVector Nz;
//    ChMatrixNM<double, 1, 3> Nx_d0;
//    ChMatrixNM<double, 1, 3> Ny_d0;
//    ChMatrixNM<double, 1, 3> Nz_d0;
//    double detJ0 = this->Calc_detJ0(0, 0, 0, Nx, Ny, Nz, Nx_d0, Ny_d0, Nz_d0);
//
//    // Transformation : Orthogonal transformation (A and J)
//    ChVector<double> G1xG2;  // Cross product of first and second column of
//    double G1dotG1;          // Dot product of first column of position vector gradient
//
//    G1xG2.x() = Nx_d0(0, 1) * Ny_d0(0, 2) - Nx_d0(0, 2) * Ny_d0(0, 1);
//    G1xG2.y() = Nx_d0(0, 2) * Ny_d0(0, 0) - Nx_d0(0, 0) * Ny_d0(0, 2);
//    G1xG2.z() = Nx_d0(0, 0) * Ny_d0(0, 1) - Nx_d0(0, 1) * Ny_d0(0, 0);
//    G1dotG1 = Nx_d0(0, 0) * Nx_d0(0, 0) + Nx_d0(0, 1) * Nx_d0(0, 1) + Nx_d0(0, 2) * Nx_d0(0, 2);
//
//    // Tangent Frame
//    ChVector<double> A1;
//    ChVector<double> A2;
//    ChVector<double> A3;
//    A1.x() = Nx_d0(0, 0);
//    A1.y() = Nx_d0(0, 1);
//    A1.z() = Nx_d0(0, 2);
//    A1 = A1 / sqrt(G1dotG1);
//    A3 = G1xG2.GetNormalized();
//    A2.Cross(A3, A1);
//
//    // Direction for orthotropic material
//    double theta = 0.0;  // Fiber angle
//    ChVector<double> AA1;
//    ChVector<double> AA2;
//    ChVector<double> AA3;
//    AA1 = A1;
//    AA2 = A2;
//    AA3 = A3;
//
//    /// Beta
//    ChMatrixNM<double, 3, 3> j0;
//    ChVector<double> j01;
//    ChVector<double> j02;
//    ChVector<double> j03;
//    // Calculates inverse of rd0 (j0) (position vector gradient: Initial Configuration)
//    j0(0, 0) = Ny_d0(0, 1) * Nz_d0(0, 2) - Nz_d0(0, 1) * Ny_d0(0, 2);
//    j0(0, 1) = Ny_d0(0, 2) * Nz_d0(0, 0) - Ny_d0(0, 0) * Nz_d0(0, 2);
//    j0(0, 2) = Ny_d0(0, 0) * Nz_d0(0, 1) - Nz_d0(0, 0) * Ny_d0(0, 1);
//    j0(1, 0) = Nz_d0(0, 1) * Nx_d0(0, 2) - Nx_d0(0, 1) * Nz_d0(0, 2);
//    j0(1, 1) = Nz_d0(0, 2) * Nx_d0(0, 0) - Nx_d0(0, 2) * Nz_d0(0, 0);
//    j0(1, 2) = Nz_d0(0, 0) * Nx_d0(0, 1) - Nz_d0(0, 1) * Nx_d0(0, 0);
//    j0(2, 0) = Nx_d0(0, 1) * Ny_d0(0, 2) - Ny_d0(0, 1) * Nx_d0(0, 2);
//    j0(2, 1) = Ny_d0(0, 0) * Nx_d0(0, 2) - Nx_d0(0, 0) * Ny_d0(0, 2);
//    j0(2, 2) = Nx_d0(0, 0) * Ny_d0(0, 1) - Ny_d0(0, 0) * Nx_d0(0, 1);
//    j0 /= detJ0;
//
//    j01[0] = j0(0, 0);
//    j02[0] = j0(1, 0);
//    j03[0] = j0(2, 0);
//    j01[1] = j0(0, 1);
//    j02[1] = j0(1, 1);
//    j03[1] = j0(2, 1);
//    j01[2] = j0(0, 2);
//    j02[2] = j0(1, 2);
//    j03[2] = j0(2, 2);
//
//    // Coefficients of contravariant transformation
//    ChVectorN<double, 9> beta;
//    beta(0) = Vdot(AA1, j01);
//    beta(1) = Vdot(AA2, j01);
//    beta(2) = Vdot(AA3, j01);
//    beta(3) = Vdot(AA1, j02);
//    beta(4) = Vdot(AA2, j02);
//    beta(5) = Vdot(AA3, j02);
//    beta(6) = Vdot(AA1, j03);
//    beta(7) = Vdot(AA2, j03);
//    beta(8) = Vdot(AA3, j03);
//
//    ChVectorN<double, 9> ddNx = m_ddT * Nx.transpose();
//    ChVectorN<double, 9> ddNy = m_ddT * Ny.transpose();
//    ChVectorN<double, 9> ddNz = m_ddT * Nz.transpose();
//
//    ChVectorN<double, 9> d0d0Nx = this->m_d0d0T * Nx.transpose();
//    ChVectorN<double, 9> d0d0Ny = this->m_d0d0T * Ny.transpose();
//    ChVectorN<double, 9> d0d0Nz = this->m_d0d0T * Nz.transpose();
//
//    // Strain component
//    ChVectorN<double, 6> strain_til;
//    strain_til(0) = 0.5 * (Nx.dot(ddNx) - Nx.dot(d0d0Nx));
//    strain_til(1) = 0.5 * (Ny.dot(ddNy) - Ny.dot(d0d0Ny));
//    strain_til(2) = Nx.dot(ddNy) - Nx.dot(d0d0Ny);
//    strain_til(3) = 0.5 * (Nz.dot(ddNz) - Nz.dot(d0d0Nz));
//    strain_til(4) = Nx.dot(ddNz) - Nx.dot(d0d0Nz);
//    strain_til(5) = Ny.dot(ddNz) - Ny.dot(d0d0Nz);
//
//    // For orthotropic material
//    ChVectorN<double, 6> strain;
//
//    strain(0) = strain_til(0) * beta(0) * beta(0) + strain_til(1) * beta(3) * beta(3) +
//                strain_til(2) * beta(0) * beta(3) + strain_til(3) * beta(6) * beta(6) +
//                strain_til(4) * beta(0) * beta(6) + strain_til(5) * beta(3) * beta(6);
//    strain(1) = strain_til(0) * beta(1) * beta(1) + strain_til(1) * beta(4) * beta(4) +
//                strain_til(2) * beta(1) * beta(4) + strain_til(3) * beta(7) * beta(7) +
//                strain_til(4) * beta(1) * beta(7) + strain_til(5) * beta(4) * beta(7);
//    strain(2) = strain_til(0) * 2.0 * beta(0) * beta(1) + strain_til(1) * 2.0 * beta(3) * beta(4) +
//                strain_til(2) * (beta(1) * beta(3) + beta(0) * beta(4)) + strain_til(3) * 2.0 * beta(6) * beta(7) +
//                strain_til(4) * (beta(1) * beta(6) + beta(0) * beta(7)) +
//                strain_til(5) * (beta(4) * beta(6) + beta(3) * beta(7));
//    strain(3) = strain_til(0) * beta(2) * beta(2) + strain_til(1) * beta(5) * beta(5) +
//                strain_til(2) * beta(2) * beta(5) + strain_til(3) * beta(8) * beta(8) +
//                strain_til(4) * beta(2) * beta(8) + strain_til(5) * beta(5) * beta(8);
//    strain(4) = strain_til(0) * 2.0 * beta(0) * beta(2) + strain_til(1) * 2.0 * beta(3) * beta(5) +
//                strain_til(2) * (beta(2) * beta(3) + beta(0) * beta(5)) + strain_til(3) * 2.0 * beta(6) * beta(8) +
//                strain_til(4) * (beta(2) * beta(6) + beta(0) * beta(8)) +
//                strain_til(5) * (beta(5) * beta(6) + beta(3) * beta(8));
//    strain(5) = strain_til(0) * 2.0 * beta(1) * beta(2) + strain_til(1) * 2.0 * beta(4) * beta(5) +
//                strain_til(2) * (beta(2) * beta(4) + beta(1) * beta(5)) + strain_til(3) * 2.0 * beta(7) * beta(8) +
//                strain_til(4) * (beta(2) * beta(7) + beta(1) * beta(8)) +
//                strain_til(5) * (beta(5) * beta(7) + beta(4) * beta(8));
//
//    return ChVector<>(strain(0), strain(1), strain(2));
//}
//
// void ChElementBeamANCF_MT35::EvaluateSectionDisplacement(const double u,
//                                                    const double v,
//                                                    ChVector<>& u_displ,
//                                                    ChVector<>& u_rotaz) {
//    // this is not a corotational element, so just do:
//    EvaluateSectionPoint(u, v, displ, u_displ);
//    u_rotaz = VNULL;  // no angles.. this is ANCF (or maybe return here the slope derivatives?)
//}

void ChElementBeamANCF_MT35::EvaluateSectionFrame(const double eta, ChVector<>& point, ChQuaternion<>& rot) {

    ChMatrixNMc<double, 9, 3> e_bar;
    ChVectorN<double, 9> Sxi_compact; 
    ChMatrixNMc<double, 9, 3> Sxi_D;

    CalcCoordMatrix(e_bar);
    Calc_Sxi_compact(Sxi_compact, eta, 0, 0);
    Calc_Sxi_D(Sxi_D, eta, 0, 0);

    // r = Se
    point = e_bar.transpose()*Sxi_compact;

    //Since ANCF does not use rotations, calculate an approximate
    //rotation based off the position vector gradients
    ChVector<double> BeamAxisTangent = e_bar.transpose()*Sxi_D.col(0);
    ChVector<double> CrossSectionY = e_bar.transpose()*Sxi_D.col(1);

    //Since the position vector gradients are not in general orthogonal,
    // set the Dx direction tangent to the beam axis and  
    // compute the Dy and Dz directions by using a
    // Gram-Schmidt orthonormalization, guided by the cross section Y direction
    ChMatrix33<> msect;
    msect.Set_A_Xdir(BeamAxisTangent, CrossSectionY);

    rot = msect.Get_A_quaternion();
}

// void ChElementBeamANCF_MT35::EvaluateSectionPoint(const double u,
//                                             const double v,
//                                             ChVector<>& point) {
//    ChVector<> u_displ;
//
//    ChMatrixNM<double, 1, 8> N;
//
//    double x = u;  // because ShapeFunctions() works in -1..1 range
//    double y = v;  // because ShapeFunctions() works in -1..1 range
//    double z = 0;
//
//    this->ShapeFunctions(N, x, y, z);
//
//    const ChVector<>& pA = m_nodes[0]->GetPos();
//    const ChVector<>& pB = m_nodes[1]->GetPos();
//    const ChVector<>& pC = m_nodes[2]->GetPos();
//    const ChVector<>& pD = m_nodes[3]->GetPos();
//
//    point.x() = N(0) * pA.x() + N(2) * pB.x() + N(4) * pC.x() + N(6) * pD.x();
//    point.y() = N(0) * pA.y() + N(2) * pB.y() + N(4) * pC.y() + N(6) * pD.y();
//    point.z() = N(0) * pA.z() + N(2) * pB.z() + N(4) * pC.z() + N(6) * pD.z();
//}

// -----------------------------------------------------------------------------
// Functions for ChLoadable interface
// -----------------------------------------------------------------------------

// Gets all the DOFs packed in a single vector (position part).
void ChElementBeamANCF_MT35::LoadableGetStateBlock_x(int block_offset, ChState& mD) {
    mD.segment(block_offset + 0, 3) = m_nodes[0]->GetPos().eigen();
    mD.segment(block_offset + 3, 3) = m_nodes[0]->GetD().eigen();
    mD.segment(block_offset + 6, 3) = m_nodes[0]->GetDD().eigen();

    mD.segment(block_offset + 9, 3) = m_nodes[1]->GetPos().eigen();
    mD.segment(block_offset + 12, 3) = m_nodes[1]->GetD().eigen();
    mD.segment(block_offset + 15, 3) = m_nodes[1]->GetDD().eigen();

    mD.segment(block_offset + 18, 3) = m_nodes[2]->GetPos().eigen();
    mD.segment(block_offset + 21, 3) = m_nodes[2]->GetD().eigen();
    mD.segment(block_offset + 24, 3) = m_nodes[2]->GetDD().eigen();
}

// Gets all the DOFs packed in a single vector (velocity part).
void ChElementBeamANCF_MT35::LoadableGetStateBlock_w(int block_offset, ChStateDelta& mD) {
    mD.segment(block_offset + 0, 3) = m_nodes[0]->GetPos_dt().eigen();
    mD.segment(block_offset + 3, 3) = m_nodes[0]->GetD_dt().eigen();
    mD.segment(block_offset + 6, 3) = m_nodes[0]->GetDD_dt().eigen();

    mD.segment(block_offset + 9, 3) = m_nodes[1]->GetPos_dt().eigen();
    mD.segment(block_offset + 12, 3) = m_nodes[1]->GetD_dt().eigen();
    mD.segment(block_offset + 15, 3) = m_nodes[1]->GetDD_dt().eigen();

    mD.segment(block_offset + 18, 3) = m_nodes[2]->GetPos_dt().eigen();
    mD.segment(block_offset + 21, 3) = m_nodes[2]->GetD_dt().eigen();
    mD.segment(block_offset + 24, 3) = m_nodes[2]->GetDD_dt().eigen();
}

/// Increment all DOFs using a delta.
void ChElementBeamANCF_MT35::LoadableStateIncrement(const unsigned int off_x,
                                               ChState& x_new,
                                               const ChState& x,
                                               const unsigned int off_v,
                                               const ChStateDelta& Dv) {
    m_nodes[0]->NodeIntStateIncrement(off_x, x_new, x, off_v, Dv);
    m_nodes[1]->NodeIntStateIncrement(off_x + 9, x_new, x, off_v + 9, Dv);
    m_nodes[2]->NodeIntStateIncrement(off_x + 18, x_new, x, off_v + 18, Dv);
}

//void ChElementBeamANCF_MT35::EvaluateSectionVelNorm(double U, ChVector<>& Result) {
//    ShapeVector N;
//    ShapeFunctions(N, U, 0, 0);
//    for (unsigned int ii = 0; ii < 3; ii++) {
//        Result += N(ii * 3) * m_nodes[ii]->GetPos_dt();
//        Result += N(ii * 3 + 1) * m_nodes[ii]->GetPos_dt();
//    }
//}

// Get the pointers to the contained ChVariables, appending to the mvars vector.
void ChElementBeamANCF_MT35::LoadableGetVariables(std::vector<ChVariables*>& mvars) {
    for (int i = 0; i < m_nodes.size(); ++i) {
        mvars.push_back(&m_nodes[i]->Variables());
        mvars.push_back(&m_nodes[i]->Variables_D());
        mvars.push_back(&m_nodes[i]->Variables_DD());
    }
}

// Evaluate N'*F , where N is the shape function evaluated at (U) coordinates of the centerline.
void ChElementBeamANCF_MT35::ComputeNF(
    const double U,              // parametric coordinate in surface
    ChVectorDynamic<>& Qi,       // Return result of Q = N'*F  here
    double& detJ,                // Return det[J] here
    const ChVectorDynamic<>& F,  // Input F vector, size is =n. field coords.
    ChVectorDynamic<>* state_x,  // if != 0, update state (pos. part) to this, then evaluate Q
    ChVectorDynamic<>* state_w   // if != 0, update state (speed part) to this, then evaluate Q
) {
    ComputeNF(U, 0, 0, Qi, detJ, F, state_x, state_w);
}


// Evaluate N'*F , where N is the shape function evaluated at (U,V,W) coordinates of the surface.
void ChElementBeamANCF_MT35::ComputeNF(
    const double U,              // parametric coordinate in volume
    const double V,              // parametric coordinate in volume
    const double W,              // parametric coordinate in volume
    ChVectorDynamic<>& Qi,       // Return result of N'*F  here, maybe with offset block_offset
    double& detJ,                // Return det[J] here
    const ChVectorDynamic<>& F,  // Input F vector, size is = n.field coords.
    ChVectorDynamic<>* state_x,  // if != 0, update state (pos. part) to this, then evaluate Q
    ChVectorDynamic<>* state_w   // if != 0, update state (speed part) to this, then evaluate Q
) {
    //Compute the generalized force vector for the applied force
    ChMatrixNM<double, 3, 27> Sxi;
    Calc_Sxi(Sxi, U, 0, 0);
    Qi = Sxi.transpose()*F.segment(0, 3);

    //Compute the generalized force vector for the applied moment
    ChMatrixNMc<double, 9, 3> e_bar;
    ChMatrixNMc<double, 9, 3> Sxi_D;
    ChMatrixNM<double, 3, 9> Sxi_D_transpose;
    ChMatrix33<double> J_Cxi;
    ChMatrix33<double> J_Cxi_Inv;
    ChVectorN<double, 9> G_A;
    ChVectorN<double, 9> G_B;
    ChVectorN<double, 9> G_C;
    ChVectorN<double, 3> M_scaled = 0.5*F.segment(3, 3);

    CalcCoordMatrix(e_bar);
    Calc_Sxi_D(Sxi_D, U, V, W);

    J_Cxi.noalias() = e_bar.transpose()*Sxi_D;
    J_Cxi_Inv = J_Cxi.inverse();

    //Compute the unique pieces that make up the moment projection matrix "G"
    //See: Antonio M Recuero, Javier F Aceituno, Jose L Escalona, and Ahmed A Shabana. 
    //A nonlinear approach for modeling rail flexibility using the absolute nodal coordinate
    //formulation. Nonlinear Dynamics, 83(1-2):463-481, 2016.
    Sxi_D_transpose = Sxi_D.transpose();
    G_A = Sxi_D_transpose.row(0)*J_Cxi_Inv(0, 0) + Sxi_D_transpose.row(1)*J_Cxi_Inv(1, 0) + Sxi_D_transpose.row(2)*J_Cxi_Inv(2, 0);
    G_B = Sxi_D_transpose.row(0)*J_Cxi_Inv(0, 1) + Sxi_D_transpose.row(1)*J_Cxi_Inv(1, 1) + Sxi_D_transpose.row(2)*J_Cxi_Inv(2, 1);
    G_C = Sxi_D_transpose.row(0)*J_Cxi_Inv(0, 2) + Sxi_D_transpose.row(1)*J_Cxi_Inv(1, 2) + Sxi_D_transpose.row(2)*J_Cxi_Inv(2, 2);

    //Compute G'M without actually forming the complete matrix "G" (since it has a sparsity pattern to it)
    //// MIKE Clean-up when slicing becomes available in Eigen 3.4
    for (unsigned int i = 0; i < 9; i++) {
        Qi(3 * i) += M_scaled(1) * G_C(i) - M_scaled(2) * G_B(i);
        Qi((3 * i) + 1) += M_scaled(2) * G_A(i) - M_scaled(0) * G_C(i);
        Qi((3 * i) + 2) += M_scaled(0) * G_B(i) - M_scaled(1) * G_A(i);
    }

    //Compute the element Jacobian between the current configuration and the normalized configuration
    //This is different than the element Jacobian between the reference configuration and the normalized
    //  configuration used in the internal force calculations
    detJ = J_Cxi.determinant();
}

// -----------------------------------------------------------------------------
//
// -----------------------------------------------------------------------------

// Calculate average element density (needed for ChLoaderVolumeGravity).
double ChElementBeamANCF_MT35::GetDensity() {
    return GetMaterial()->Get_rho();
}

// Calculate tangent to the centerline at (U) coordinates.
ChVector<> ChElementBeamANCF_MT35::ComputeTangent(const double U) {
    ChMatrixNMc<double, 9, 3> e_bar;
    ChMatrixNMc<double, 9, 3> Sxi_D;
    ChVector<> r_xi;

    CalcCoordMatrix(e_bar);
    Calc_Sxi_D(Sxi_D, U, 0, 0);
    r_xi = e_bar.transpose()*Sxi_D.col(1);

    return r_xi.GetNormalized();
}

////////////////////////////////////////////////////////////////

//#ifndef CH_QUADRATURE_STATIC_TABLES
#define CH_QUADRATURE_STATIC_TABLES 10
ChQuadratureTables static_tables_MT35(1, CH_QUADRATURE_STATIC_TABLES);
//#endif // !CH_QUADRATURE_STATIC_TABLES

ChQuadratureTables* ChElementBeamANCF_MT35::GetStaticGQTables() {
    return &static_tables_MT35;
}

////////////////////////////////////////////////////////////////

// ============================================================================
// Implementation of ChMaterialBeamANCF_MT35 methods
// ============================================================================

// Construct an isotropic material.
ChMaterialBeamANCF_MT35::ChMaterialBeamANCF_MT35(double rho,        // material density
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
ChMaterialBeamANCF_MT35::ChMaterialBeamANCF_MT35(double rho,            // material density
                                       const ChVector<>& E,   // elasticity moduli (E_x, E_y, E_z)
                                       const ChVector<>& nu,  // Poisson ratios (nu_xy, nu_xz, nu_yz)
                                       const ChVector<>& G,   // shear moduli (G_xy, G_xz, G_yz)
                                       const double& k1,      // Shear correction factor along beam local y axis
                                       const double& k2       // Shear correction factor along beam local z axis
                                       )
    : m_rho(rho) {
    Calc_D0_Dv(E, nu, G, k1, k2);
}

//Calculate the matrix form of two stiffness tensors used by the ANCF beam for selective reduced integration of the Poisson effect
void ChMaterialBeamANCF_MT35::Calc_D0_Dv(const ChVector<>& E, const ChVector<>& nu, const ChVector<>& G, double k1, double k2) {

    //orthotropic material ref: http://homes.civil.aau.dk/lda/Continuum/material.pdf
    //except position of the shear terms is different to match the original ANCF reference paper

    double nu_12 = nu.x();
    double nu_13 = nu.y();
    double nu_23 = nu.z();
    double nu_21 = nu_12 * E.y() / E.x();
    double nu_31 = nu_13 * E.z() / E.x();
    double nu_32 = nu_23 * E.z() / E.y();
    double k = 1.0 - nu_23*nu_32 - nu_12*nu_21 - nu_13*nu_31 - nu_12*nu_23*nu_31 - nu_21*nu_32*nu_13;

    //Component of Stiffness Tensor that does not contain the Poisson Effect
    m_D0(0) = E.x();
    m_D0(1) = E.y();
    m_D0(2) = E.z();
    m_D0(3) = G.z();
    m_D0(4) = G.y()*k1;
    m_D0(5) = G.x()*k2;

    //Remaining components of the Stiffness Tensor that contain the Poisson Effect 
    m_Dv(0, 0) = E.x()*(1 - nu_23*nu_32) / k - m_D0(0);
    m_Dv(1, 0) = E.y()*(nu_13*nu_32 + nu_12) / k;
    m_Dv(2, 0) = E.z()*(nu_12*nu_23 + nu_13) / k;

    m_Dv(0, 1) = E.x()*(nu_23*nu_31 + nu_21) / k;
    m_Dv(1, 1) = E.y()*(1 - nu_13*nu_31) / k - m_D0(1);
    m_Dv(2, 1) = E.z()*(nu_13*nu_21 + nu_23) / k;

    m_Dv(0, 2) = E.x()*(nu_21*nu_32 + nu_31) / k;
    m_Dv(1, 2) = E.y()*(nu_12*nu_31 + nu_32) / k;
    m_Dv(2, 2) = E.z()*(1 - nu_12*nu_21) / k - m_D0(2);
}

}  // end of namespace fea
}  // end of namespace chrono

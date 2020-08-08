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
// Authors: Mike Taylor
// =============================================================================
//
// Checking the internal force calculation with intrinsics outside of an element 
//
// =============================================================================

//#include "mkl.h"

#include <immintrin.h>
#include <string>
//#include "unsupported/Eigen/FFT"

#include "chrono/ChConfig.h"
#include "chrono/utils/ChBenchmark.h"

#include "chrono/physics/ChSystemSMC.h"
#include "chrono/solver/ChIterativeSolverLS.h"

#include "chrono_postprocess/ChGnuPlot.h"
#include "chrono_thirdparty/filesystem/path.h"

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
#include "chrono/fea/ChElementBeamANCF_MT60.h"
#include "chrono/fea/ChElementBeamANCF_MT61.h"
#include "chrono/fea/ChElementBeamANCF_MT62.h"

#include "chrono/fea/ChMesh.h"
#include "chrono/fea/ChVisualizationFEAmesh.h"
#include "chrono/physics/ChLoadContainer.h"
#include "chrono/fea/ChLoadsBeam.h"

#ifdef CHRONO_IRRLICHT
#include "chrono_irrlicht/ChIrrApp.h"
#endif

#ifdef CHRONO_MKL
#include "chrono_mkl/ChSolverMKL.h"
#endif

#ifdef CHRONO_MUMPS
#include "chrono_mumps/ChSolverMumps.h"
#endif

using namespace chrono;
using namespace chrono::fea;
using namespace postprocess;

// =============================================================================

/// Dense matrix with *fixed size* (known at compile time).
/// A ChMatrixNM is templated by the type of its coefficients and by the matrix dimensions (number of rows and columns).
template <typename T, int M, int N>
using ChMatrixNMc = Eigen::Matrix<T, M, N, Eigen::ColMajor>;

#define CH_QUADRATURE_STATIC_TABLES 10
ChQuadratureTables static_GQ_tables(1, CH_QUADRATURE_STATIC_TABLES);

// =============================================================================

void Calc_Sxi_D(ChMatrixNMc<double, 9, 3>& Sxi_D, double xi, double eta, double zeta, double thicknessY, double thicknessZ) {
    Sxi_D(0, 0) = xi - 0.5;
    Sxi_D(1, 0) = 0.25*thicknessY*eta*(2.0*xi - 1.0);
    Sxi_D(2, 0) = 0.25*thicknessZ*zeta*(2.0*xi - 1.0);
    Sxi_D(3, 0) = xi + 0.5;
    Sxi_D(4, 0) = 0.25*thicknessY*eta*(2.0*xi + 1.0);
    Sxi_D(5, 0) = 0.25*thicknessZ*zeta*(2.0*xi + 1.0);
    Sxi_D(6, 0) = -2.0*xi;
    Sxi_D(7, 0) = -thicknessY * eta*xi;
    Sxi_D(8, 0) = -thicknessZ * zeta*xi;

    Sxi_D(0, 1) = 0.0;
    Sxi_D(1, 1) = 0.25*thicknessY*(xi*xi - xi);
    Sxi_D(2, 1) = 0.0;
    Sxi_D(3, 1) = 0.0;
    Sxi_D(4, 1) = 0.25*thicknessY*(xi*xi + xi);
    Sxi_D(5, 1) = 0.0;
    Sxi_D(6, 1) = 0.0;
    Sxi_D(7, 1) = 0.5*thicknessY*(1 - xi * xi);
    Sxi_D(8, 1) = 0.0;

    Sxi_D(0, 2) = 0.0;
    Sxi_D(1, 2) = 0.0;
    Sxi_D(2, 2) = 0.25*thicknessZ*(xi*xi - xi);
    Sxi_D(3, 2) = 0.0;
    Sxi_D(4, 2) = 0.0;
    Sxi_D(5, 2) = 0.25*thicknessZ*(xi*xi + xi);
    Sxi_D(6, 2) = 0.0;
    Sxi_D(7, 2) = 0.0;
    Sxi_D(8, 2) = 0.5*thicknessZ*(1 - xi * xi);
}

//Precalculate constant matrices and scalars for the internal force calculations
void PrecomputeInternalForceMatricesWeights(ChMatrixNMc<double, 9, 3>& e0_bar,
                                            double thicknessY,
                                            double thicknessZ,
                                            ChVectorN<double, 16>& GQWeight_det_J_0xi_D0,
                                            ChMatrixNMc<double, 9, 16>& SD_precompute_D0_col0_block,
                                            ChMatrixNMc<double, 9, 16>& SD_precompute_D0_col1_block,
                                            ChMatrixNMc<double, 9, 16>& SD_precompute_D0_col2_block,
                                            ChVectorN<double, 4>& GQWeight_det_J_0xi_Dv,
                                            ChMatrixNMc<double, 9, 4>& SD_precompute_Dv_col0_block,
                                            ChMatrixNMc<double, 9, 4>& SD_precompute_Dv_col1_block,
                                            ChMatrixNMc<double, 9, 4>& SD_precompute_Dv_col2_block,
											ChMatrixNMc<double, 9, 60>& SD_precompute_col_ordered,
											ChVectorN<double, 20>& GQWeight_det_J_0xi){
    ChQuadratureTables* GQTable = &static_GQ_tables;
    unsigned int GQ_idx_xi = 3;       // 4 Point Gauss-Quadrature;
    unsigned int GQ_idx_eta_zeta = 1; // 2 Point Gauss-Quadrature;

    ChMatrixNMc<double, 9, 60> SD_precompute;             ///< Precomputed corrected normalized shape function derivative matrices for no Poisson Effect followed by Poisson Effect on the beam axis only
    //ChVectorN<double, 20> GQWeight_det_J_0xi;             ///< Precomputed Gauss-Quadrature Weight & Element Jacobian scale factors for no Poisson Effect followed by Poisson Effect on the beam axis only

    ChMatrixNMc<double, 9, 48> SD_precompute_D0;          ///< Precomputed corrected normalized shape function derivative matrices for no Poisson Effect
    //ChVectorN<double, 16> GQWeight_det_J_0xi_D0;          ///< Precomputed Gauss-Quadrature Weight & Element Jacobian scale factors for no Poisson Effect
    ChMatrixNMc<double, 9, 12> SD_precompute_Dv;           ///< Precomputed corrected normalized shape function derivative matrices for Poisson Effect on the beam axis only
    //ChVectorN<double, 4> GQWeight_det_J_0xi_Dv;           ///< Precomputed Gauss-Quadrature Weight & Element Jacobian scale factor for Poisson Effect on the beam axis only

    //ChMatrixNMc<double, 9, 16> SD_precompute_D0_col0_block;
    //ChMatrixNMc<double, 9, 16> SD_precompute_D0_col1_block;
    //ChMatrixNMc<double, 9, 16> SD_precompute_D0_col2_block;

    //Precalculate the matrices of normalized shape function derivatives corrected for a potentially non-straight reference configuration & GQ Weights times the determiate of the element Jacobian for later
    //Calculating the portion of the Selective Reduced Integration that does account for the Poisson effect
    for (unsigned int it_xi = 0; it_xi < GQTable->Lroots[GQ_idx_xi].size(); it_xi++) {
        for (unsigned int it_eta = 0; it_eta < GQTable->Lroots[GQ_idx_eta_zeta].size(); it_eta++) {
            for (unsigned int it_zeta = 0; it_zeta < GQTable->Lroots[GQ_idx_eta_zeta].size(); it_zeta++) {
                double GQ_weight = GQTable->Weight[GQ_idx_xi][it_xi] * GQTable->Weight[GQ_idx_eta_zeta][it_eta] * GQTable->Weight[GQ_idx_eta_zeta][it_zeta];
                double xi = GQTable->Lroots[GQ_idx_xi][it_xi];
                double eta = GQTable->Lroots[GQ_idx_eta_zeta][it_eta];
                double zeta = GQTable->Lroots[GQ_idx_eta_zeta][it_zeta];
                auto index = it_zeta + it_eta * GQTable->Lroots[GQ_idx_eta_zeta].size() + it_xi * GQTable->Lroots[GQ_idx_eta_zeta].size()*GQTable->Lroots[GQ_idx_eta_zeta].size();
                ChMatrix33<double> J_0xi;               //Element Jacobian between the reference configuration and normalized configuration
                ChMatrixNMc<double, 9, 3> Sxi_D;         //Matrix of normalized shape function derivatives

                Calc_Sxi_D(Sxi_D, xi, eta, zeta, thicknessY, thicknessZ);
                J_0xi.noalias() = e0_bar.transpose()*Sxi_D;

                ChMatrixNMc<double, 9, 3> SD = Sxi_D * J_0xi.inverse();
                SD_precompute_D0.block(0, 3 * index, 9, 3) = SD;

                SD_precompute_D0_col0_block.block(0, index, 9, 1) = SD.col(0);
                SD_precompute_D0_col1_block.block(0, index, 9, 1) = SD.col(1);
                SD_precompute_D0_col2_block.block(0, index, 9, 1) = SD.col(2);

                GQWeight_det_J_0xi_D0(index) = -J_0xi.determinant()*GQ_weight;
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

        Calc_Sxi_D(Sxi_D, xi, 0, 0, thicknessY, thicknessZ);
        J_0xi.noalias() = e0_bar.transpose()*Sxi_D;

        ChMatrixNMc<double, 9, 3> SD = Sxi_D * J_0xi.inverse();
        SD_precompute_Dv.block(0, 3 * it_xi, 9, 3) = SD;

        SD_precompute_Dv_col0_block.block(0, it_xi, 9, 1) = SD.col(0);
        SD_precompute_Dv_col1_block.block(0, it_xi, 9, 1) = SD.col(1);
        SD_precompute_Dv_col2_block.block(0, it_xi, 9, 1) = SD.col(2);

        GQWeight_det_J_0xi_Dv(it_xi) = -J_0xi.determinant()*GQ_weight;
    }

    SD_precompute.block(0, 0, 9, 48) = SD_precompute_D0;
    SD_precompute.block(0, 48, 9, 12) = SD_precompute_Dv;
    
	SD_precompute_col_ordered.block(0, 0, 9, 16) = SD_precompute_D0_col0_block;
    SD_precompute_col_ordered.block(0, 16, 9, 16) = SD_precompute_D0_col1_block;
    SD_precompute_col_ordered.block(0, 32, 9, 16) = SD_precompute_D0_col2_block;
    SD_precompute_col_ordered.block(0, 48, 9, 4) = SD_precompute_Dv_col0_block;
    SD_precompute_col_ordered.block(0, 52, 9, 4) = SD_precompute_Dv_col1_block;
    SD_precompute_col_ordered.block(0, 56, 9, 4) = SD_precompute_Dv_col2_block;

    GQWeight_det_J_0xi.block(0, 0, 16, 1) = GQWeight_det_J_0xi_D0;
    GQWeight_det_J_0xi.block(16, 0, 4, 1) = GQWeight_det_J_0xi_Dv;
}

void Check_No_Damping_Internal_Force(ChMatrixNMc<double, 9, 3>& e0_bar, ChMatrixNMc<double, 9, 3>& e_bar) {
    double L = 1;   //m
    double W = 0.1; //m
    double H = 0.1; //m
    double E = 210e9; //Pa
    double nu = 0.3;
    double k2 = 10 * (1 + nu) / (12 + 11 * nu);  // Timoshenko shear correction coefficient for a rectangular cross - section
    double k3 = k2;
    double G = E / (2 * (1 + nu));
    ChVectorN<double, 6> D0;
    D0 << E, E, E, G, G*k2, G*k3;

    ChMatrixNM<double, 3, 3> Dv;
    Dv << 2 * nu, 1, 1,
        1, 2 * nu, 1,
        1, 1, 2 * nu;
    Dv *= E * nu / ((1 + nu)*(1 - 2 * nu));

    //ChMatrixNMc<double, 9, 3> e0_bar;
    //e0_bar << 0, 0, 0,
    //    0, 1, 0,
    //    0, 0, 1,
    //    1, 0, 0,
    //    0, 1, 0,
    //    0, 0, 1,
    //    0.5, 0, 0,
    //    0, 1, 0,
    //    0, 0, 1;

    //ChMatrixNMc<double, 9, 3> e_bar;
    //e_bar << 0, 0, 0,
    //    0, 1, 0,
    //    0, 0, 1,
    //    1, 0, 0,
    //    0, 1, 0,
    //    0, 0, 1,
    //    0.5, 0, 0.001,
    //    0, 1, 0,
    //    0, 0, 1;

    ChVectorN<double, 16> GQWeight_det_J_0xi_D0;
    ChMatrixNMc<double, 9, 16> SD_precompute_D0_col0_block;
    ChMatrixNMc<double, 9, 16> SD_precompute_D0_col1_block;
    ChMatrixNMc<double, 9, 16> SD_precompute_D0_col2_block;
    ChVectorN<double, 4> GQWeight_det_J_0xi_Dv;
    ChMatrixNMc<double, 9, 4> SD_precompute_Dv_col0_block;
    ChMatrixNMc<double, 9, 4> SD_precompute_Dv_col1_block;
    ChMatrixNMc<double, 9, 4> SD_precompute_Dv_col2_block;
    ChMatrixNMc<double, 9, 60> SD_precompute_col_ordered;
    ChVectorN<double, 20> GQWeight_det_J_0xi;

    PrecomputeInternalForceMatricesWeights(e0_bar, W, H, GQWeight_det_J_0xi_D0,
        SD_precompute_D0_col0_block, SD_precompute_D0_col1_block, SD_precompute_D0_col2_block,
        GQWeight_det_J_0xi_Dv, SD_precompute_Dv_col0_block, SD_precompute_Dv_col1_block, SD_precompute_Dv_col2_block,
        SD_precompute_col_ordered, GQWeight_det_J_0xi);

    // =============================================================================

    ChMatrixNMc<double, 16, 3> Fcol0_Transpose_D0_Block = SD_precompute_D0_col0_block.transpose()*e_bar;
    ChMatrixNMc<double, 16, 3> Fcol1_Transpose_D0_Block = SD_precompute_D0_col1_block.transpose()*e_bar;
    ChMatrixNMc<double, 16, 3> Fcol2_Transpose_D0_Block = SD_precompute_D0_col2_block.transpose()*e_bar;

    ChVectorN<double, 16> SPK2_0_D0_Block = 0.5*D0(0)*GQWeight_det_J_0xi_D0.array()*(Fcol0_Transpose_D0_Block.rowwise().squaredNorm().array() - 1);
    ChVectorN<double, 16> SPK2_1_D0_Block = 0.5*D0(1)*GQWeight_det_J_0xi_D0.array()*(Fcol1_Transpose_D0_Block.rowwise().squaredNorm().array() - 1);
    ChVectorN<double, 16> SPK2_2_D0_Block = 0.5*D0(2)*GQWeight_det_J_0xi_D0.array()*(Fcol2_Transpose_D0_Block.rowwise().squaredNorm().array() - 1);
    ChVectorN<double, 16> SPK2_3_D0_Block = D0(3)*GQWeight_det_J_0xi_D0.cwiseProduct(Fcol1_Transpose_D0_Block.cwiseProduct(Fcol2_Transpose_D0_Block).rowwise().sum());
    ChVectorN<double, 16> SPK2_4_D0_Block = D0(4)*GQWeight_det_J_0xi_D0.cwiseProduct(Fcol0_Transpose_D0_Block.cwiseProduct(Fcol2_Transpose_D0_Block).rowwise().sum());
    ChVectorN<double, 16> SPK2_5_D0_Block = D0(5)*GQWeight_det_J_0xi_D0.cwiseProduct(Fcol0_Transpose_D0_Block.cwiseProduct(Fcol1_Transpose_D0_Block).rowwise().sum());

    ChMatrixNM<double, 16, 3> Ptransposed_scaled_row0_D0_Block =
        Fcol0_Transpose_D0_Block.array().colwise()*SPK2_0_D0_Block.array() +
        Fcol1_Transpose_D0_Block.array().colwise()*SPK2_5_D0_Block.array() +
        Fcol2_Transpose_D0_Block.array().colwise()*SPK2_4_D0_Block.array();

    ChMatrixNM<double, 16, 3> Ptransposed_scaled_row1_D0_Block =
        Fcol0_Transpose_D0_Block.array().colwise()*SPK2_5_D0_Block.array() +
        Fcol1_Transpose_D0_Block.array().colwise()*SPK2_1_D0_Block.array() +
        Fcol2_Transpose_D0_Block.array().colwise()*SPK2_3_D0_Block.array();

    ChMatrixNM<double, 16, 3> Ptransposed_scaled_row2_D0_Block =
        Fcol0_Transpose_D0_Block.array().colwise()*SPK2_4_D0_Block.array() +
        Fcol1_Transpose_D0_Block.array().colwise()*SPK2_3_D0_Block.array() +
        Fcol2_Transpose_D0_Block.array().colwise()*SPK2_2_D0_Block.array();

    // =============================================================================

    ChMatrixNMc<double, 4, 3> Fcol0_Transpose_Dv_Block = SD_precompute_Dv_col0_block.transpose()*e_bar;
    ChMatrixNMc<double, 4, 3> Fcol1_Transpose_Dv_Block = SD_precompute_Dv_col1_block.transpose()*e_bar;
    ChMatrixNMc<double, 4, 3> Fcol2_Transpose_Dv_Block = SD_precompute_Dv_col2_block.transpose()*e_bar;

    ChVectorN<double, 4> Ediag_0_Dv_Block = 0.5 * GQWeight_det_J_0xi_Dv.array() * (Fcol0_Transpose_Dv_Block.rowwise().squaredNorm().array() - 1);
    ChVectorN<double, 4> Ediag_1_Dv_Block = 0.5 * GQWeight_det_J_0xi_Dv.array() * (Fcol1_Transpose_Dv_Block.rowwise().squaredNorm().array() - 1);
    ChVectorN<double, 4> Ediag_2_Dv_Block = 0.5 * GQWeight_det_J_0xi_Dv.array() * (Fcol2_Transpose_Dv_Block.rowwise().squaredNorm().array() - 1);

    ChVectorN<double, 4> Sdiag_0_Dv_Block = Dv(0, 0)*Ediag_0_Dv_Block + Dv(1, 0)*Ediag_1_Dv_Block + Dv(2, 0)*Ediag_2_Dv_Block;
    ChVectorN<double, 4> Sdiag_1_Dv_Block = Dv(0, 1)*Ediag_0_Dv_Block + Dv(1, 1)*Ediag_1_Dv_Block + Dv(2, 1)*Ediag_2_Dv_Block;
    ChVectorN<double, 4> Sdiag_2_Dv_Block = Dv(0, 2)*Ediag_0_Dv_Block + Dv(1, 2)*Ediag_1_Dv_Block + Dv(2, 2)*Ediag_2_Dv_Block;

    ChMatrixNM<double, 4, 3> Ptransposed_scaled_row0_Dv_Block =
        Fcol0_Transpose_Dv_Block.array().colwise()*Sdiag_0_Dv_Block.array();
    ChMatrixNM<double, 4, 3> Ptransposed_scaled_row1_Dv_Block =
        Fcol1_Transpose_Dv_Block.array().colwise()*Sdiag_1_Dv_Block.array();
    ChMatrixNM<double, 4, 3> Ptransposed_scaled_row2_Dv_Block =
        Fcol2_Transpose_Dv_Block.array().colwise()*Sdiag_2_Dv_Block.array();

    // =============================================================================

    ChMatrixNM<double, 9, 3> QiCompact =
        SD_precompute_D0_col0_block * Ptransposed_scaled_row0_D0_Block +
        SD_precompute_D0_col1_block * Ptransposed_scaled_row1_D0_Block +
        SD_precompute_D0_col2_block * Ptransposed_scaled_row2_D0_Block +
        SD_precompute_Dv_col0_block * Ptransposed_scaled_row0_Dv_Block +
        SD_precompute_Dv_col1_block * Ptransposed_scaled_row1_Dv_Block +
        SD_precompute_Dv_col2_block * Ptransposed_scaled_row2_Dv_Block;
    Eigen::Map<ChVectorN<double, 27>> QiReshaped(QiCompact.data(), QiCompact.size());
    ChVectorN<double, 27> Fi = QiReshaped;

    std::cout << Fi << std::endl;

}


void Check_Damping_Internal_Force(ChMatrixNMc<double, 9, 3>& e0_bar, ChMatrixNMc<double, 9, 3>& e_bar, ChMatrixNMc<double, 9, 3>& edot_bar) {
    double L = 1;   //m
    double W = 0.1; //m
    double H = 0.1; //m
    double E = 210e9; //Pa
    double nu = 0.3;
    double k2 = 10 * (1 + nu) / (12 + 11 * nu);  // Timoshenko shear correction coefficient for a rectangular cross - section
    double k3 = k2;
    double G = E / (2 * (1 + nu));

    double alpha = 0.01;

    ChVectorN<double, 6> D0;
    D0 << E, E, E, G, G*k2, G*k3;

    ChMatrixNM<double, 3, 3> Dv;
    Dv << 2 * nu, 1, 1,
        1, 2 * nu, 1,
        1, 1, 2 * nu;
    Dv *= E * nu / ((1 + nu)*(1 - 2 * nu));

    ChMatrixNMc<double, 9, 6> ebar_ebardot;
    ebar_ebardot.block(0, 0, 9, 3) = e_bar;
    ebar_ebardot.block(0, 3, 9, 3) = edot_bar;

    //ChMatrixNMc<double, 9, 3> e0_bar;
    //e0_bar << 0, 0, 0,
    //    0, 1, 0,
    //    0, 0, 1,
    //    1, 0, 0,
    //    0, 1, 0,
    //    0, 0, 1,
    //    0.5, 0, 0,
    //    0, 1, 0,
    //    0, 0, 1;

    //ChMatrixNMc<double, 9, 3> e_bar;
    //e_bar << 0, 0, 0,
    //    0, 1, 0,
    //    0, 0, 1,
    //    1, 0, 0,
    //    0, 1, 0,
    //    0, 0, 1,
    //    0.5, 0, 0.001,
    //    0, 1, 0,
    //    0, 0, 1;

    ChVectorN<double, 16> GQWeight_det_J_0xi_D0;
    ChMatrixNMc<double, 9, 16> SD_precompute_D0_col0_block;
    ChMatrixNMc<double, 9, 16> SD_precompute_D0_col1_block;
    ChMatrixNMc<double, 9, 16> SD_precompute_D0_col2_block;
    ChVectorN<double, 4> GQWeight_det_J_0xi_Dv;
    ChMatrixNMc<double, 9, 4> SD_precompute_Dv_col0_block;
    ChMatrixNMc<double, 9, 4> SD_precompute_Dv_col1_block;
    ChMatrixNMc<double, 9, 4> SD_precompute_Dv_col2_block;
	ChMatrixNMc<double, 9, 60> SD_precompute_col_ordered;
    ChVectorN<double, 20> GQWeight_det_J_0xi_temp;

    PrecomputeInternalForceMatricesWeights(e0_bar, W, H, GQWeight_det_J_0xi_D0,
        SD_precompute_D0_col0_block, SD_precompute_D0_col1_block, SD_precompute_D0_col2_block,
        GQWeight_det_J_0xi_Dv, SD_precompute_Dv_col0_block, SD_precompute_Dv_col1_block, SD_precompute_Dv_col2_block,
		SD_precompute_col_ordered, GQWeight_det_J_0xi_temp);

    // =============================================================================


    double* GQWeight_det_J_0xi_Pointer = static_cast<double*>(_mm_malloc(sizeof(double) * 20, 32));
    Eigen::Map<ChVectorN<double, 20>> GQWeight_det_J_0xi(GQWeight_det_J_0xi_Pointer);
    GQWeight_det_J_0xi = GQWeight_det_J_0xi_temp;

	//Calculate F is one big block and then split up afterwards to improve efficiency (hopefully)
	//double* F_Transpose_CombinedBlock_col_ordered_Pointer = (double*)aligned_alloc(32, 360 * sizeof(double));
    double* F_Transpose_CombinedBlock_col_ordered_Pointer = static_cast<double*>(_mm_malloc(sizeof(double) * 360, 32));
    Eigen::Map<ChMatrixNMc<double, 60, 6>> F_Transpose_CombinedBlock_col_ordered(F_Transpose_CombinedBlock_col_ordered_Pointer);
    F_Transpose_CombinedBlock_col_ordered = SD_precompute_col_ordered.transpose()*ebar_ebardot;

    double* P_transpose_scaled_Block_col_ordered_Pointer = static_cast<double*>(_mm_malloc(sizeof(double) * 180, 32));
    Eigen::Map<ChMatrixNMc<double, 60, 3>> P_transpose_scaled_Block_col_ordered(P_transpose_scaled_Block_col_ordered_Pointer);
    P_transpose_scaled_Block_col_ordered.setZero();

    //std::cout << F_Transpose_CombinedBlock_col_ordered.block(0, 0, 4, 6) << std::endl;
    //std::printf("%lg %lg %lg %lg %lg %lg\n", 
    //    F_Transpose_CombinedBlock_col_ordered_Pointer[0],
    //    F_Transpose_CombinedBlock_col_ordered_Pointer[60],
    //    F_Transpose_CombinedBlock_col_ordered_Pointer[120],
    //    F_Transpose_CombinedBlock_col_ordered_Pointer[180],
    //    F_Transpose_CombinedBlock_col_ordered_Pointer[240],
    //    F_Transpose_CombinedBlock_col_ordered_Pointer[300]);
    //std::printf("%lf %lf %lf %lf %lf %lf\n",
    //    F_Transpose_CombinedBlock_col_ordered_Pointer[1],
    //    F_Transpose_CombinedBlock_col_ordered_Pointer[61],
    //    F_Transpose_CombinedBlock_col_ordered_Pointer[121],
    //    F_Transpose_CombinedBlock_col_ordered_Pointer[181],
    //    F_Transpose_CombinedBlock_col_ordered_Pointer[241],
    //    F_Transpose_CombinedBlock_col_ordered_Pointer[301]);
    //std::printf("%lf %lf %lf %lf %lf %lf\n",
    //    F_Transpose_CombinedBlock_col_ordered_Pointer[2],
    //    F_Transpose_CombinedBlock_col_ordered_Pointer[62],
    //    F_Transpose_CombinedBlock_col_ordered_Pointer[122],
    //    F_Transpose_CombinedBlock_col_ordered_Pointer[182],
    //    F_Transpose_CombinedBlock_col_ordered_Pointer[242],
    //    F_Transpose_CombinedBlock_col_ordered_Pointer[302]);
    //std::printf("%lf %lf %lf %lf %lf %lf\n",
    //    F_Transpose_CombinedBlock_col_ordered_Pointer[3],
    //    F_Transpose_CombinedBlock_col_ordered_Pointer[63],
    //    F_Transpose_CombinedBlock_col_ordered_Pointer[123],
    //    F_Transpose_CombinedBlock_col_ordered_Pointer[183],
    //    F_Transpose_CombinedBlock_col_ordered_Pointer[243],
    //    F_Transpose_CombinedBlock_col_ordered_Pointer[303]);


//-------------------------------------------------------------
    
    __m256d ones = _mm256_set1_pd(1.0);
    __m256d Alpha = _mm256_set1_pd(alpha);
    __m256d TwiceAlpha = _mm256_set1_pd(2.0*alpha);

//-------------------------------------------------------------

    for (auto i = 0; i < 16; i += 4) {

        __m256d GQWeight_det_J_0xi_AVX = _mm256_loadu_pd(&GQWeight_det_J_0xi[i]);

        //-------------------------------------------------------------

        __m256d Fcol0_Transpose_D0_Block_col0 = _mm256_load_pd(&F_Transpose_CombinedBlock_col_ordered_Pointer[i]);
        __m256d Fcol0_Transpose_D0_Block_col1 = _mm256_load_pd(&F_Transpose_CombinedBlock_col_ordered_Pointer[60 + i]);
        __m256d Fcol0_Transpose_D0_Block_col2 = _mm256_load_pd(&F_Transpose_CombinedBlock_col_ordered_Pointer[120 + i]);
        __m256d Fdotcol0_Transpose_D0_Block_col0 = _mm256_load_pd(&F_Transpose_CombinedBlock_col_ordered_Pointer[180 + i]);
        __m256d Fdotcol0_Transpose_D0_Block_col1 = _mm256_load_pd(&F_Transpose_CombinedBlock_col_ordered_Pointer[240 + i]);
        __m256d Fdotcol0_Transpose_D0_Block_col2 = _mm256_load_pd(&F_Transpose_CombinedBlock_col_ordered_Pointer[300 + i]);

        __m256d SPK2_Scale = _mm256_set1_pd(0.5 * D0(0));

        __m256d SPK2_0_D0_Block = _mm256_mul_pd(Fcol0_Transpose_D0_Block_col0, Fcol0_Transpose_D0_Block_col0);
        SPK2_0_D0_Block = _mm256_fmadd_pd(Fcol0_Transpose_D0_Block_col1, Fcol0_Transpose_D0_Block_col1, SPK2_0_D0_Block);
        SPK2_0_D0_Block = _mm256_fmadd_pd(Fcol0_Transpose_D0_Block_col2, Fcol0_Transpose_D0_Block_col2, SPK2_0_D0_Block);
        SPK2_0_D0_Block = _mm256_sub_pd(SPK2_0_D0_Block, ones);
        __m256d BlockDamping = _mm256_mul_pd(Fcol0_Transpose_D0_Block_col0, Fdotcol0_Transpose_D0_Block_col0);
        BlockDamping = _mm256_fmadd_pd(Fcol0_Transpose_D0_Block_col1, Fdotcol0_Transpose_D0_Block_col1, BlockDamping);
        BlockDamping = _mm256_fmadd_pd(Fcol0_Transpose_D0_Block_col2, Fdotcol0_Transpose_D0_Block_col2, BlockDamping);
        SPK2_0_D0_Block = _mm256_fmadd_pd(TwiceAlpha, BlockDamping, SPK2_0_D0_Block);
        SPK2_0_D0_Block = _mm256_mul_pd(SPK2_0_D0_Block, GQWeight_det_J_0xi_AVX);
        SPK2_0_D0_Block = _mm256_mul_pd(SPK2_0_D0_Block, SPK2_Scale);

        //-------------------------------------------------------------

        __m256d Fcol1_Transpose_D0_Block_col0 = _mm256_load_pd(&F_Transpose_CombinedBlock_col_ordered_Pointer[16 + i]);
        __m256d Fcol1_Transpose_D0_Block_col1 = _mm256_load_pd(&F_Transpose_CombinedBlock_col_ordered_Pointer[76 + i]);
        __m256d Fcol1_Transpose_D0_Block_col2 = _mm256_load_pd(&F_Transpose_CombinedBlock_col_ordered_Pointer[136 + i]);
        __m256d Fdotcol1_Transpose_D0_Block_col0 = _mm256_load_pd(&F_Transpose_CombinedBlock_col_ordered_Pointer[196 + i]);
        __m256d Fdotcol1_Transpose_D0_Block_col1 = _mm256_load_pd(&F_Transpose_CombinedBlock_col_ordered_Pointer[256 + i]);
        __m256d Fdotcol1_Transpose_D0_Block_col2 = _mm256_load_pd(&F_Transpose_CombinedBlock_col_ordered_Pointer[316 + i]);

        SPK2_Scale = _mm256_set1_pd(0.5 * D0(1));

        __m256d SPK2_1_D0_Block = _mm256_mul_pd(Fcol1_Transpose_D0_Block_col0, Fcol1_Transpose_D0_Block_col0);
        SPK2_1_D0_Block = _mm256_fmadd_pd(Fcol1_Transpose_D0_Block_col1, Fcol1_Transpose_D0_Block_col1, SPK2_1_D0_Block);
        SPK2_1_D0_Block = _mm256_fmadd_pd(Fcol1_Transpose_D0_Block_col2, Fcol1_Transpose_D0_Block_col2, SPK2_1_D0_Block);
        SPK2_1_D0_Block = _mm256_sub_pd(SPK2_1_D0_Block, ones);
        BlockDamping = _mm256_mul_pd(Fcol1_Transpose_D0_Block_col0, Fdotcol1_Transpose_D0_Block_col0);
        BlockDamping = _mm256_fmadd_pd(Fcol1_Transpose_D0_Block_col1, Fdotcol1_Transpose_D0_Block_col1, BlockDamping);
        BlockDamping = _mm256_fmadd_pd(Fcol1_Transpose_D0_Block_col2, Fdotcol1_Transpose_D0_Block_col2, BlockDamping);
        SPK2_1_D0_Block = _mm256_fmadd_pd(TwiceAlpha, BlockDamping, SPK2_1_D0_Block);
        SPK2_1_D0_Block = _mm256_mul_pd(SPK2_1_D0_Block, GQWeight_det_J_0xi_AVX);
        SPK2_1_D0_Block = _mm256_mul_pd(SPK2_1_D0_Block, SPK2_Scale);

        //-------------------------------------------------------------

        __m256d Fcol2_Transpose_D0_Block_col0 = _mm256_load_pd(&F_Transpose_CombinedBlock_col_ordered_Pointer[32 + i]);
        __m256d Fcol2_Transpose_D0_Block_col1 = _mm256_load_pd(&F_Transpose_CombinedBlock_col_ordered_Pointer[92 + i]);
        __m256d Fcol2_Transpose_D0_Block_col2 = _mm256_load_pd(&F_Transpose_CombinedBlock_col_ordered_Pointer[152 + i]);
        __m256d Fdotcol2_Transpose_D0_Block_col0 = _mm256_load_pd(&F_Transpose_CombinedBlock_col_ordered_Pointer[212 + i]);
        __m256d Fdotcol2_Transpose_D0_Block_col1 = _mm256_load_pd(&F_Transpose_CombinedBlock_col_ordered_Pointer[272 + i]);
        __m256d Fdotcol2_Transpose_D0_Block_col2 = _mm256_load_pd(&F_Transpose_CombinedBlock_col_ordered_Pointer[332 + i]);

        SPK2_Scale = _mm256_set1_pd(0.5 * D0(2));

        __m256d SPK2_2_D0_Block = _mm256_mul_pd(Fcol2_Transpose_D0_Block_col0, Fcol2_Transpose_D0_Block_col0);
        SPK2_2_D0_Block = _mm256_fmadd_pd(Fcol2_Transpose_D0_Block_col1, Fcol2_Transpose_D0_Block_col1, SPK2_2_D0_Block);
        SPK2_2_D0_Block = _mm256_fmadd_pd(Fcol2_Transpose_D0_Block_col2, Fcol2_Transpose_D0_Block_col2, SPK2_2_D0_Block);
        SPK2_2_D0_Block = _mm256_sub_pd(SPK2_2_D0_Block, ones);
        BlockDamping = _mm256_mul_pd(Fcol2_Transpose_D0_Block_col0, Fdotcol2_Transpose_D0_Block_col0);
        BlockDamping = _mm256_fmadd_pd(Fcol2_Transpose_D0_Block_col1, Fdotcol2_Transpose_D0_Block_col1, BlockDamping);
        BlockDamping = _mm256_fmadd_pd(Fcol2_Transpose_D0_Block_col2, Fdotcol2_Transpose_D0_Block_col2, BlockDamping);
        SPK2_2_D0_Block = _mm256_fmadd_pd(TwiceAlpha, BlockDamping, SPK2_2_D0_Block);
        SPK2_2_D0_Block = _mm256_mul_pd(SPK2_2_D0_Block, GQWeight_det_J_0xi_AVX);
        SPK2_2_D0_Block = _mm256_mul_pd(SPK2_2_D0_Block, SPK2_Scale);

        //-------------------------------------------------------------

        SPK2_Scale = _mm256_set1_pd(D0(3));

        __m256d SPK2_3_D0_Block = _mm256_mul_pd(Fcol1_Transpose_D0_Block_col0, Fcol2_Transpose_D0_Block_col0);
        SPK2_3_D0_Block = _mm256_fmadd_pd(Fcol1_Transpose_D0_Block_col1, Fcol2_Transpose_D0_Block_col1, SPK2_3_D0_Block);
        SPK2_3_D0_Block = _mm256_fmadd_pd(Fcol1_Transpose_D0_Block_col2, Fcol2_Transpose_D0_Block_col2, SPK2_3_D0_Block);
        BlockDamping = _mm256_mul_pd(Fcol2_Transpose_D0_Block_col0, Fdotcol1_Transpose_D0_Block_col0);
        BlockDamping = _mm256_fmadd_pd(Fcol2_Transpose_D0_Block_col1, Fdotcol1_Transpose_D0_Block_col1, BlockDamping);
        BlockDamping = _mm256_fmadd_pd(Fcol2_Transpose_D0_Block_col2, Fdotcol1_Transpose_D0_Block_col2, BlockDamping);
        BlockDamping = _mm256_fmadd_pd(Fcol1_Transpose_D0_Block_col0, Fdotcol2_Transpose_D0_Block_col0, BlockDamping);
        BlockDamping = _mm256_fmadd_pd(Fcol1_Transpose_D0_Block_col1, Fdotcol2_Transpose_D0_Block_col1, BlockDamping);
        BlockDamping = _mm256_fmadd_pd(Fcol1_Transpose_D0_Block_col2, Fdotcol2_Transpose_D0_Block_col2, BlockDamping);
        SPK2_3_D0_Block = _mm256_fmadd_pd(Alpha, BlockDamping, SPK2_3_D0_Block);
        SPK2_3_D0_Block = _mm256_mul_pd(SPK2_3_D0_Block, GQWeight_det_J_0xi_AVX);
        SPK2_3_D0_Block = _mm256_mul_pd(SPK2_3_D0_Block, SPK2_Scale);

        //-------------------------------------------------------------

        SPK2_Scale = _mm256_set1_pd(D0(4));

        __m256d SPK2_4_D0_Block = _mm256_mul_pd(Fcol0_Transpose_D0_Block_col0, Fcol2_Transpose_D0_Block_col0);
        SPK2_4_D0_Block = _mm256_fmadd_pd(Fcol0_Transpose_D0_Block_col1, Fcol2_Transpose_D0_Block_col1, SPK2_4_D0_Block);
        SPK2_4_D0_Block = _mm256_fmadd_pd(Fcol0_Transpose_D0_Block_col2, Fcol2_Transpose_D0_Block_col2, SPK2_4_D0_Block);
        BlockDamping = _mm256_mul_pd(Fcol2_Transpose_D0_Block_col0, Fdotcol0_Transpose_D0_Block_col0);
        BlockDamping = _mm256_fmadd_pd(Fcol2_Transpose_D0_Block_col1, Fdotcol0_Transpose_D0_Block_col1, BlockDamping);
        BlockDamping = _mm256_fmadd_pd(Fcol2_Transpose_D0_Block_col2, Fdotcol0_Transpose_D0_Block_col2, BlockDamping);
        BlockDamping = _mm256_fmadd_pd(Fcol0_Transpose_D0_Block_col0, Fdotcol2_Transpose_D0_Block_col0, BlockDamping);
        BlockDamping = _mm256_fmadd_pd(Fcol0_Transpose_D0_Block_col1, Fdotcol2_Transpose_D0_Block_col1, BlockDamping);
        BlockDamping = _mm256_fmadd_pd(Fcol0_Transpose_D0_Block_col2, Fdotcol2_Transpose_D0_Block_col2, BlockDamping);
        SPK2_4_D0_Block = _mm256_fmadd_pd(Alpha, BlockDamping, SPK2_4_D0_Block);
        SPK2_4_D0_Block = _mm256_mul_pd(SPK2_4_D0_Block, GQWeight_det_J_0xi_AVX);
        SPK2_4_D0_Block = _mm256_mul_pd(SPK2_4_D0_Block, SPK2_Scale);

        //-------------------------------------------------------------

        SPK2_Scale = _mm256_set1_pd(D0(5));

        __m256d SPK2_5_D0_Block = _mm256_mul_pd(Fcol0_Transpose_D0_Block_col0, Fcol1_Transpose_D0_Block_col0);
        SPK2_5_D0_Block = _mm256_fmadd_pd(Fcol0_Transpose_D0_Block_col1, Fcol1_Transpose_D0_Block_col1, SPK2_5_D0_Block);
        SPK2_5_D0_Block = _mm256_fmadd_pd(Fcol0_Transpose_D0_Block_col2, Fcol1_Transpose_D0_Block_col2, SPK2_5_D0_Block);
        BlockDamping = _mm256_mul_pd(Fcol1_Transpose_D0_Block_col0, Fdotcol0_Transpose_D0_Block_col0);
        BlockDamping = _mm256_fmadd_pd(Fcol1_Transpose_D0_Block_col1, Fdotcol0_Transpose_D0_Block_col1, BlockDamping);
        BlockDamping = _mm256_fmadd_pd(Fcol1_Transpose_D0_Block_col2, Fdotcol0_Transpose_D0_Block_col2, BlockDamping);
        BlockDamping = _mm256_fmadd_pd(Fcol0_Transpose_D0_Block_col0, Fdotcol1_Transpose_D0_Block_col0, BlockDamping);
        BlockDamping = _mm256_fmadd_pd(Fcol0_Transpose_D0_Block_col1, Fdotcol1_Transpose_D0_Block_col1, BlockDamping);
        BlockDamping = _mm256_fmadd_pd(Fcol0_Transpose_D0_Block_col2, Fdotcol1_Transpose_D0_Block_col2, BlockDamping);
        SPK2_5_D0_Block = _mm256_fmadd_pd(Alpha, BlockDamping, SPK2_5_D0_Block);
        SPK2_5_D0_Block = _mm256_mul_pd(SPK2_5_D0_Block, GQWeight_det_J_0xi_AVX);
        SPK2_5_D0_Block = _mm256_mul_pd(SPK2_5_D0_Block, SPK2_Scale);

        //-------------------------------------------------------------

        __m256d P_transpose_scaled_Block = _mm256_mul_pd(Fcol0_Transpose_D0_Block_col0, SPK2_0_D0_Block);
        P_transpose_scaled_Block = _mm256_fmadd_pd(Fcol1_Transpose_D0_Block_col0, SPK2_5_D0_Block, P_transpose_scaled_Block);
        P_transpose_scaled_Block = _mm256_fmadd_pd(Fcol2_Transpose_D0_Block_col0, SPK2_4_D0_Block, P_transpose_scaled_Block);
        _mm256_store_pd(&P_transpose_scaled_Block_col_ordered_Pointer[i], P_transpose_scaled_Block);

        P_transpose_scaled_Block = _mm256_mul_pd(Fcol0_Transpose_D0_Block_col1, SPK2_0_D0_Block);
        P_transpose_scaled_Block = _mm256_fmadd_pd(Fcol1_Transpose_D0_Block_col1, SPK2_5_D0_Block, P_transpose_scaled_Block);
        P_transpose_scaled_Block = _mm256_fmadd_pd(Fcol2_Transpose_D0_Block_col1, SPK2_4_D0_Block, P_transpose_scaled_Block);
        _mm256_store_pd(&P_transpose_scaled_Block_col_ordered_Pointer[60 + i], P_transpose_scaled_Block);

        P_transpose_scaled_Block = _mm256_mul_pd(Fcol0_Transpose_D0_Block_col2, SPK2_0_D0_Block);
        P_transpose_scaled_Block = _mm256_fmadd_pd(Fcol1_Transpose_D0_Block_col2, SPK2_5_D0_Block, P_transpose_scaled_Block);
        P_transpose_scaled_Block = _mm256_fmadd_pd(Fcol2_Transpose_D0_Block_col2, SPK2_4_D0_Block, P_transpose_scaled_Block);
        _mm256_store_pd(&P_transpose_scaled_Block_col_ordered_Pointer[120 + i], P_transpose_scaled_Block);


        P_transpose_scaled_Block = _mm256_mul_pd(Fcol0_Transpose_D0_Block_col0, SPK2_5_D0_Block);
        P_transpose_scaled_Block = _mm256_fmadd_pd(Fcol1_Transpose_D0_Block_col0, SPK2_1_D0_Block, P_transpose_scaled_Block);
        P_transpose_scaled_Block = _mm256_fmadd_pd(Fcol2_Transpose_D0_Block_col0, SPK2_3_D0_Block, P_transpose_scaled_Block);
        _mm256_store_pd(&P_transpose_scaled_Block_col_ordered_Pointer[16 + i], P_transpose_scaled_Block);

        P_transpose_scaled_Block = _mm256_mul_pd(Fcol0_Transpose_D0_Block_col1, SPK2_5_D0_Block);
        P_transpose_scaled_Block = _mm256_fmadd_pd(Fcol1_Transpose_D0_Block_col1, SPK2_1_D0_Block, P_transpose_scaled_Block);
        P_transpose_scaled_Block = _mm256_fmadd_pd(Fcol2_Transpose_D0_Block_col1, SPK2_3_D0_Block, P_transpose_scaled_Block);
        _mm256_store_pd(&P_transpose_scaled_Block_col_ordered_Pointer[76 + i], P_transpose_scaled_Block);

        P_transpose_scaled_Block = _mm256_mul_pd(Fcol0_Transpose_D0_Block_col2, SPK2_5_D0_Block);
        P_transpose_scaled_Block = _mm256_fmadd_pd(Fcol1_Transpose_D0_Block_col2, SPK2_1_D0_Block, P_transpose_scaled_Block);
        P_transpose_scaled_Block = _mm256_fmadd_pd(Fcol2_Transpose_D0_Block_col2, SPK2_3_D0_Block, P_transpose_scaled_Block);
        _mm256_store_pd(&P_transpose_scaled_Block_col_ordered_Pointer[136 + i], P_transpose_scaled_Block);


        P_transpose_scaled_Block = _mm256_mul_pd(Fcol0_Transpose_D0_Block_col0, SPK2_4_D0_Block);
        P_transpose_scaled_Block = _mm256_fmadd_pd(Fcol1_Transpose_D0_Block_col0, SPK2_3_D0_Block, P_transpose_scaled_Block);
        P_transpose_scaled_Block = _mm256_fmadd_pd(Fcol2_Transpose_D0_Block_col0, SPK2_2_D0_Block, P_transpose_scaled_Block);
        _mm256_store_pd(&P_transpose_scaled_Block_col_ordered_Pointer[32 + i], P_transpose_scaled_Block);

        P_transpose_scaled_Block = _mm256_mul_pd(Fcol0_Transpose_D0_Block_col1, SPK2_4_D0_Block);
        P_transpose_scaled_Block = _mm256_fmadd_pd(Fcol1_Transpose_D0_Block_col1, SPK2_3_D0_Block, P_transpose_scaled_Block);
        P_transpose_scaled_Block = _mm256_fmadd_pd(Fcol2_Transpose_D0_Block_col1, SPK2_2_D0_Block, P_transpose_scaled_Block);
        _mm256_store_pd(&P_transpose_scaled_Block_col_ordered_Pointer[92 + i], P_transpose_scaled_Block);

        P_transpose_scaled_Block = _mm256_mul_pd(Fcol0_Transpose_D0_Block_col2, SPK2_4_D0_Block);
        P_transpose_scaled_Block = _mm256_fmadd_pd(Fcol1_Transpose_D0_Block_col2, SPK2_3_D0_Block, P_transpose_scaled_Block);
        P_transpose_scaled_Block = _mm256_fmadd_pd(Fcol2_Transpose_D0_Block_col2, SPK2_2_D0_Block, P_transpose_scaled_Block);
        _mm256_store_pd(&P_transpose_scaled_Block_col_ordered_Pointer[152 + i], P_transpose_scaled_Block);
    }

    {
        __m256d GQWeight_det_J_0xi_AVX = _mm256_loadu_pd(&GQWeight_det_J_0xi[16]);
        __m256d halves = _mm256_set1_pd(0.5);

        //-------------------------------------------------------------

        __m256d Fcol0_Transpose_D0_Block_col0 = _mm256_load_pd(&F_Transpose_CombinedBlock_col_ordered_Pointer[48]);
        __m256d Fcol0_Transpose_D0_Block_col1 = _mm256_load_pd(&F_Transpose_CombinedBlock_col_ordered_Pointer[108]);
        __m256d Fcol0_Transpose_D0_Block_col2 = _mm256_load_pd(&F_Transpose_CombinedBlock_col_ordered_Pointer[168]);
        __m256d Fdotcol0_Transpose_D0_Block_col0 = _mm256_load_pd(&F_Transpose_CombinedBlock_col_ordered_Pointer[228]);
        __m256d Fdotcol0_Transpose_D0_Block_col1 = _mm256_load_pd(&F_Transpose_CombinedBlock_col_ordered_Pointer[288]);
        __m256d Fdotcol0_Transpose_D0_Block_col2 = _mm256_load_pd(&F_Transpose_CombinedBlock_col_ordered_Pointer[348]);

        __m256d Ediag_0_Dv_Block = _mm256_mul_pd(Fcol0_Transpose_D0_Block_col0, Fcol0_Transpose_D0_Block_col0);
        Ediag_0_Dv_Block = _mm256_fmadd_pd(Fcol0_Transpose_D0_Block_col1, Fcol0_Transpose_D0_Block_col1, Ediag_0_Dv_Block);
        Ediag_0_Dv_Block = _mm256_fmadd_pd(Fcol0_Transpose_D0_Block_col2, Fcol0_Transpose_D0_Block_col2, Ediag_0_Dv_Block);
        Ediag_0_Dv_Block = _mm256_fmsub_pd(Ediag_0_Dv_Block, halves, halves);
        __m256d BlockDamping = _mm256_mul_pd(Fcol0_Transpose_D0_Block_col0, Fdotcol0_Transpose_D0_Block_col0);
        BlockDamping = _mm256_fmadd_pd(Fcol0_Transpose_D0_Block_col1, Fdotcol0_Transpose_D0_Block_col1, BlockDamping);
        BlockDamping = _mm256_fmadd_pd(Fcol0_Transpose_D0_Block_col2, Fdotcol0_Transpose_D0_Block_col2, BlockDamping);
        Ediag_0_Dv_Block = _mm256_fmadd_pd(Alpha, BlockDamping, Ediag_0_Dv_Block);
        Ediag_0_Dv_Block = _mm256_mul_pd(Ediag_0_Dv_Block, GQWeight_det_J_0xi_AVX);

        //-------------------------------------------------------------

        __m256d Fcol1_Transpose_D0_Block_col0 = _mm256_load_pd(&F_Transpose_CombinedBlock_col_ordered_Pointer[52]);
        __m256d Fcol1_Transpose_D0_Block_col1 = _mm256_load_pd(&F_Transpose_CombinedBlock_col_ordered_Pointer[112]);
        __m256d Fcol1_Transpose_D0_Block_col2 = _mm256_load_pd(&F_Transpose_CombinedBlock_col_ordered_Pointer[172]);
        __m256d Fdotcol1_Transpose_D0_Block_col0 = _mm256_load_pd(&F_Transpose_CombinedBlock_col_ordered_Pointer[232]);
        __m256d Fdotcol1_Transpose_D0_Block_col1 = _mm256_load_pd(&F_Transpose_CombinedBlock_col_ordered_Pointer[292]);
        __m256d Fdotcol1_Transpose_D0_Block_col2 = _mm256_load_pd(&F_Transpose_CombinedBlock_col_ordered_Pointer[352]);

        __m256d Ediag_1_Dv_Block = _mm256_mul_pd(Fcol1_Transpose_D0_Block_col0, Fcol1_Transpose_D0_Block_col0);
        Ediag_1_Dv_Block = _mm256_fmadd_pd(Fcol1_Transpose_D0_Block_col1, Fcol1_Transpose_D0_Block_col1, Ediag_1_Dv_Block);
        Ediag_1_Dv_Block = _mm256_fmadd_pd(Fcol1_Transpose_D0_Block_col2, Fcol1_Transpose_D0_Block_col2, Ediag_1_Dv_Block);
        Ediag_1_Dv_Block = _mm256_fmsub_pd(Ediag_1_Dv_Block, halves, halves);
        BlockDamping = _mm256_mul_pd(Fcol1_Transpose_D0_Block_col0, Fdotcol1_Transpose_D0_Block_col0);
        BlockDamping = _mm256_fmadd_pd(Fcol1_Transpose_D0_Block_col1, Fdotcol1_Transpose_D0_Block_col1, BlockDamping);
        BlockDamping = _mm256_fmadd_pd(Fcol1_Transpose_D0_Block_col2, Fdotcol1_Transpose_D0_Block_col2, BlockDamping);
        Ediag_1_Dv_Block = _mm256_fmadd_pd(Alpha, BlockDamping, Ediag_1_Dv_Block);
        Ediag_1_Dv_Block = _mm256_mul_pd(Ediag_1_Dv_Block, GQWeight_det_J_0xi_AVX);

        //-------------------------------------------------------------

        __m256d Fcol2_Transpose_D0_Block_col0 = _mm256_load_pd(&F_Transpose_CombinedBlock_col_ordered_Pointer[56]);
        __m256d Fcol2_Transpose_D0_Block_col1 = _mm256_load_pd(&F_Transpose_CombinedBlock_col_ordered_Pointer[116]);
        __m256d Fcol2_Transpose_D0_Block_col2 = _mm256_load_pd(&F_Transpose_CombinedBlock_col_ordered_Pointer[176]);
        __m256d Fdotcol2_Transpose_D0_Block_col0 = _mm256_load_pd(&F_Transpose_CombinedBlock_col_ordered_Pointer[236]);
        __m256d Fdotcol2_Transpose_D0_Block_col1 = _mm256_load_pd(&F_Transpose_CombinedBlock_col_ordered_Pointer[296]);
        __m256d Fdotcol2_Transpose_D0_Block_col2 = _mm256_load_pd(&F_Transpose_CombinedBlock_col_ordered_Pointer[356]);

        __m256d Ediag_2_Dv_Block = _mm256_mul_pd(Fcol2_Transpose_D0_Block_col0, Fcol2_Transpose_D0_Block_col0);
        Ediag_2_Dv_Block = _mm256_fmadd_pd(Fcol2_Transpose_D0_Block_col1, Fcol2_Transpose_D0_Block_col1, Ediag_2_Dv_Block);
        Ediag_2_Dv_Block = _mm256_fmadd_pd(Fcol2_Transpose_D0_Block_col2, Fcol2_Transpose_D0_Block_col2, Ediag_2_Dv_Block);
        Ediag_2_Dv_Block = _mm256_fmsub_pd(Ediag_2_Dv_Block, halves, halves);
        BlockDamping = _mm256_mul_pd(Fcol2_Transpose_D0_Block_col0, Fdotcol2_Transpose_D0_Block_col0);
        BlockDamping = _mm256_fmadd_pd(Fcol2_Transpose_D0_Block_col1, Fdotcol2_Transpose_D0_Block_col1, BlockDamping);
        BlockDamping = _mm256_fmadd_pd(Fcol2_Transpose_D0_Block_col2, Fdotcol2_Transpose_D0_Block_col2, BlockDamping);
        Ediag_2_Dv_Block = _mm256_fmadd_pd(Alpha, BlockDamping, Ediag_2_Dv_Block);
        Ediag_2_Dv_Block = _mm256_mul_pd(Ediag_2_Dv_Block, GQWeight_det_J_0xi_AVX);

        //-------------------------------------------------------------

        __m256d Dv00 = _mm256_set1_pd(Dv(0, 0));
        __m256d Dv01 = _mm256_set1_pd(Dv(0, 1));
        __m256d Dv02 = _mm256_set1_pd(Dv(0, 2));
        __m256d Dv11 = _mm256_set1_pd(Dv(1, 1));
        __m256d Dv12 = _mm256_set1_pd(Dv(1, 2));
        __m256d Dv22 = _mm256_set1_pd(Dv(2, 2));

        __m256d Sdiag_0_Dv_Block = _mm256_mul_pd(Dv00, Ediag_0_Dv_Block);
        Sdiag_0_Dv_Block = _mm256_fmadd_pd(Dv01, Ediag_1_Dv_Block, Sdiag_0_Dv_Block);
        Sdiag_0_Dv_Block = _mm256_fmadd_pd(Dv02, Ediag_2_Dv_Block, Sdiag_0_Dv_Block);

        __m256d Sdiag_1_Dv_Block = _mm256_mul_pd(Dv01, Ediag_0_Dv_Block);
        Sdiag_1_Dv_Block = _mm256_fmadd_pd(Dv11, Ediag_1_Dv_Block, Sdiag_1_Dv_Block);
        Sdiag_1_Dv_Block = _mm256_fmadd_pd(Dv12, Ediag_2_Dv_Block, Sdiag_1_Dv_Block);

        __m256d Sdiag_2_Dv_Block = _mm256_mul_pd(Dv02, Ediag_0_Dv_Block);
        Sdiag_2_Dv_Block = _mm256_fmadd_pd(Dv12, Ediag_1_Dv_Block, Sdiag_2_Dv_Block);
        Sdiag_2_Dv_Block = _mm256_fmadd_pd(Dv22, Ediag_2_Dv_Block, Sdiag_2_Dv_Block);

        //-------------------------------------------------------------

        __m256d P_transpose_scaled_Block = _mm256_mul_pd(Fcol0_Transpose_D0_Block_col0, Sdiag_0_Dv_Block);
        _mm256_store_pd(&P_transpose_scaled_Block_col_ordered_Pointer[48], P_transpose_scaled_Block);

        P_transpose_scaled_Block = _mm256_mul_pd(Fcol0_Transpose_D0_Block_col1, Sdiag_0_Dv_Block);
        _mm256_store_pd(&P_transpose_scaled_Block_col_ordered_Pointer[108], P_transpose_scaled_Block);

        P_transpose_scaled_Block = _mm256_mul_pd(Fcol0_Transpose_D0_Block_col2, Sdiag_0_Dv_Block);
        _mm256_store_pd(&P_transpose_scaled_Block_col_ordered_Pointer[168], P_transpose_scaled_Block);


        P_transpose_scaled_Block = _mm256_mul_pd(Fcol1_Transpose_D0_Block_col0, Sdiag_1_Dv_Block);
        _mm256_store_pd(&P_transpose_scaled_Block_col_ordered_Pointer[52], P_transpose_scaled_Block);

        P_transpose_scaled_Block = _mm256_mul_pd(Fcol1_Transpose_D0_Block_col1, Sdiag_1_Dv_Block);
        _mm256_store_pd(&P_transpose_scaled_Block_col_ordered_Pointer[112], P_transpose_scaled_Block);

        P_transpose_scaled_Block = _mm256_mul_pd(Fcol1_Transpose_D0_Block_col2, Sdiag_1_Dv_Block);
        _mm256_store_pd(&P_transpose_scaled_Block_col_ordered_Pointer[172], P_transpose_scaled_Block);


        P_transpose_scaled_Block = _mm256_mul_pd(Fcol2_Transpose_D0_Block_col0, Sdiag_2_Dv_Block);
        _mm256_store_pd(&P_transpose_scaled_Block_col_ordered_Pointer[56], P_transpose_scaled_Block);

        P_transpose_scaled_Block = _mm256_mul_pd(Fcol2_Transpose_D0_Block_col1, Sdiag_2_Dv_Block);
        _mm256_store_pd(&P_transpose_scaled_Block_col_ordered_Pointer[116], P_transpose_scaled_Block);

        P_transpose_scaled_Block = _mm256_mul_pd(Fcol2_Transpose_D0_Block_col2, Sdiag_2_Dv_Block);
        _mm256_store_pd(&P_transpose_scaled_Block_col_ordered_Pointer[176], P_transpose_scaled_Block);
    }

    std::cout << F_Transpose_CombinedBlock_col_ordered << std::endl;
    ChMatrixNMc<double, 60, 6> F_transpose_scaled_Block_col_ordered_Debug = F_Transpose_CombinedBlock_col_ordered;
    ChMatrixNMc<double, 60, 3> P_transpose_scaled_Block_col_ordered_Debug = P_transpose_scaled_Block_col_ordered;

    ChMatrixNM<double, 9, 3> QiCompact2 = SD_precompute_col_ordered * P_transpose_scaled_Block_col_ordered;
    Eigen::Map<ChVectorN<double, 27>> QiReshaped2(QiCompact2.data(), QiCompact2.size());
    ChVectorN<double, 27> Fi2 = QiReshaped2;



    ChMatrixNMc<double, 16, 3> Fcol0_Transpose_D0_Block = SD_precompute_D0_col0_block.transpose()*e_bar;
    ChMatrixNMc<double, 16, 3> Fcol1_Transpose_D0_Block = SD_precompute_D0_col1_block.transpose()*e_bar;
    ChMatrixNMc<double, 16, 3> Fcol2_Transpose_D0_Block = SD_precompute_D0_col2_block.transpose()*e_bar;
    ChMatrixNMc<double, 16, 3> Fdotcol0_Transpose_D0_Block = SD_precompute_D0_col0_block.transpose()*edot_bar;
    ChMatrixNMc<double, 16, 3> Fdotcol1_Transpose_D0_Block = SD_precompute_D0_col1_block.transpose()*edot_bar;
    ChMatrixNMc<double, 16, 3> Fdotcol2_Transpose_D0_Block = SD_precompute_D0_col2_block.transpose()*edot_bar;

    ChVectorN<double, 16> SPK2_0_D0_Block = D0(0)*GQWeight_det_J_0xi_D0.array()*(0.5*Fcol0_Transpose_D0_Block.rowwise().squaredNorm().array() - 0.5 + alpha * Fcol0_Transpose_D0_Block.cwiseProduct(Fdotcol0_Transpose_D0_Block).rowwise().sum().array());
    ChVectorN<double, 16> SPK2_1_D0_Block = D0(1)*GQWeight_det_J_0xi_D0.array()*(0.5*Fcol1_Transpose_D0_Block.rowwise().squaredNorm().array() - 0.5 + alpha * Fcol1_Transpose_D0_Block.cwiseProduct(Fdotcol1_Transpose_D0_Block).rowwise().sum().array());
    ChVectorN<double, 16> SPK2_2_D0_Block = D0(2)*GQWeight_det_J_0xi_D0.array()*(0.5*Fcol2_Transpose_D0_Block.rowwise().squaredNorm().array() - 0.5 + alpha * Fcol2_Transpose_D0_Block.cwiseProduct(Fdotcol2_Transpose_D0_Block).rowwise().sum().array());
    ChVectorN<double, 16> SPK2_3_D0_Block = D0(3)*GQWeight_det_J_0xi_D0.cwiseProduct(Fcol1_Transpose_D0_Block.cwiseProduct(Fcol2_Transpose_D0_Block).rowwise().sum() + alpha * (Fcol1_Transpose_D0_Block.cwiseProduct(Fdotcol2_Transpose_D0_Block).rowwise().sum() + Fcol2_Transpose_D0_Block.cwiseProduct(Fdotcol1_Transpose_D0_Block).rowwise().sum()));
    ChVectorN<double, 16> SPK2_4_D0_Block = D0(4)*GQWeight_det_J_0xi_D0.cwiseProduct(Fcol0_Transpose_D0_Block.cwiseProduct(Fcol2_Transpose_D0_Block).rowwise().sum() + alpha * (Fcol0_Transpose_D0_Block.cwiseProduct(Fdotcol2_Transpose_D0_Block).rowwise().sum() + Fcol2_Transpose_D0_Block.cwiseProduct(Fdotcol0_Transpose_D0_Block).rowwise().sum()));
    ChVectorN<double, 16> SPK2_5_D0_Block = D0(5)*GQWeight_det_J_0xi_D0.cwiseProduct(Fcol0_Transpose_D0_Block.cwiseProduct(Fcol1_Transpose_D0_Block).rowwise().sum() + alpha * (Fcol0_Transpose_D0_Block.cwiseProduct(Fdotcol1_Transpose_D0_Block).rowwise().sum() + Fcol1_Transpose_D0_Block.cwiseProduct(Fdotcol0_Transpose_D0_Block).rowwise().sum()));

    ChMatrixNM<double, 16, 3> Ptransposed_scaled_row0_D0_Block =
        Fcol0_Transpose_D0_Block.array().colwise()*SPK2_0_D0_Block.array() +
        Fcol1_Transpose_D0_Block.array().colwise()*SPK2_5_D0_Block.array() +
        Fcol2_Transpose_D0_Block.array().colwise()*SPK2_4_D0_Block.array();

    ChMatrixNM<double, 16, 3> Ptransposed_scaled_row1_D0_Block =
        Fcol0_Transpose_D0_Block.array().colwise()*SPK2_5_D0_Block.array() +
        Fcol1_Transpose_D0_Block.array().colwise()*SPK2_1_D0_Block.array() +
        Fcol2_Transpose_D0_Block.array().colwise()*SPK2_3_D0_Block.array();

    ChMatrixNM<double, 16, 3> Ptransposed_scaled_row2_D0_Block =
        Fcol0_Transpose_D0_Block.array().colwise()*SPK2_4_D0_Block.array() +
        Fcol1_Transpose_D0_Block.array().colwise()*SPK2_3_D0_Block.array() +
        Fcol2_Transpose_D0_Block.array().colwise()*SPK2_2_D0_Block.array();

    // =============================================================================

    ChMatrixNMc<double, 4, 3> Fcol0_Transpose_Dv_Block = SD_precompute_Dv_col0_block.transpose()*e_bar;
    ChMatrixNMc<double, 4, 3> Fcol1_Transpose_Dv_Block = SD_precompute_Dv_col1_block.transpose()*e_bar;
    ChMatrixNMc<double, 4, 3> Fcol2_Transpose_Dv_Block = SD_precompute_Dv_col2_block.transpose()*e_bar;
    ChMatrixNMc<double, 4, 3> Fdotcol0_Transpose_Dv_Block = SD_precompute_Dv_col0_block.transpose()*edot_bar;
    ChMatrixNMc<double, 4, 3> Fdotcol1_Transpose_Dv_Block = SD_precompute_Dv_col1_block.transpose()*edot_bar;
    ChMatrixNMc<double, 4, 3> Fdotcol2_Transpose_Dv_Block = SD_precompute_Dv_col2_block.transpose()*edot_bar;

    ChVectorN<double, 4> Ediag_0_Dv_Block = GQWeight_det_J_0xi_Dv.array() * (0.5*Fcol0_Transpose_Dv_Block.rowwise().squaredNorm().array() - 0.5 + alpha * Fcol0_Transpose_Dv_Block.cwiseProduct(Fdotcol0_Transpose_Dv_Block).rowwise().sum().array());
    ChVectorN<double, 4> Ediag_1_Dv_Block = GQWeight_det_J_0xi_Dv.array() * (0.5*Fcol1_Transpose_Dv_Block.rowwise().squaredNorm().array() - 0.5 + alpha * Fcol1_Transpose_Dv_Block.cwiseProduct(Fdotcol1_Transpose_Dv_Block).rowwise().sum().array());
    ChVectorN<double, 4> Ediag_2_Dv_Block = GQWeight_det_J_0xi_Dv.array() * (0.5*Fcol2_Transpose_Dv_Block.rowwise().squaredNorm().array() - 0.5 + alpha * Fcol2_Transpose_Dv_Block.cwiseProduct(Fdotcol2_Transpose_Dv_Block).rowwise().sum().array());

    ChVectorN<double, 4> Sdiag_0_Dv_Block = Dv(0, 0)*Ediag_0_Dv_Block + Dv(1, 0)*Ediag_1_Dv_Block + Dv(2, 0)*Ediag_2_Dv_Block;
    ChVectorN<double, 4> Sdiag_1_Dv_Block = Dv(0, 1)*Ediag_0_Dv_Block + Dv(1, 1)*Ediag_1_Dv_Block + Dv(2, 1)*Ediag_2_Dv_Block;
    ChVectorN<double, 4> Sdiag_2_Dv_Block = Dv(0, 2)*Ediag_0_Dv_Block + Dv(1, 2)*Ediag_1_Dv_Block + Dv(2, 2)*Ediag_2_Dv_Block;

    ChMatrixNM<double, 4, 3> Ptransposed_scaled_row0_Dv_Block =
        Fcol0_Transpose_Dv_Block.array().colwise()*Sdiag_0_Dv_Block.array();
    ChMatrixNM<double, 4, 3> Ptransposed_scaled_row1_Dv_Block =
        Fcol1_Transpose_Dv_Block.array().colwise()*Sdiag_1_Dv_Block.array();
    ChMatrixNM<double, 4, 3> Ptransposed_scaled_row2_Dv_Block =
        Fcol2_Transpose_Dv_Block.array().colwise()*Sdiag_2_Dv_Block.array();

    ChMatrixNMc<double, 60, 3> P_transpose_scaled_Block_col_ordered_Debug2;
    P_transpose_scaled_Block_col_ordered_Debug2.block(0, 0, 16, 3) = Ptransposed_scaled_row0_D0_Block;
    P_transpose_scaled_Block_col_ordered_Debug2.block(16, 0, 16, 3) = Ptransposed_scaled_row1_D0_Block;
    P_transpose_scaled_Block_col_ordered_Debug2.block(32, 0, 16, 3) = Ptransposed_scaled_row2_D0_Block;
    P_transpose_scaled_Block_col_ordered_Debug2.block(48, 0, 4, 3) = Ptransposed_scaled_row0_Dv_Block;
    P_transpose_scaled_Block_col_ordered_Debug2.block(52, 0, 4, 3) = Ptransposed_scaled_row1_Dv_Block;
    P_transpose_scaled_Block_col_ordered_Debug2.block(56, 0, 4, 3) = Ptransposed_scaled_row2_Dv_Block;


    // =============================================================================

    ChMatrixNM<double, 9, 3> QiCompact =
        SD_precompute_D0_col0_block * Ptransposed_scaled_row0_D0_Block +
        SD_precompute_D0_col1_block * Ptransposed_scaled_row1_D0_Block +
        SD_precompute_D0_col2_block * Ptransposed_scaled_row2_D0_Block +
        SD_precompute_Dv_col0_block * Ptransposed_scaled_row0_Dv_Block +
        SD_precompute_Dv_col1_block * Ptransposed_scaled_row1_Dv_Block +
        SD_precompute_Dv_col2_block * Ptransposed_scaled_row2_Dv_Block;
    Eigen::Map<ChVectorN<double, 27>> QiReshaped(QiCompact.data(), QiCompact.size());
    ChVectorN<double, 27> Fi = QiReshaped;

    std::cout << Fi << std::endl;

    ChMatrixNMc<double, 27, 2> Fis;
    Fis.block(0, 0, 27, 1) = Fi;
    Fis.block(0, 1, 27, 1) = Fi2;

    std::cout << Fis << std::endl;

    _mm_free(F_Transpose_CombinedBlock_col_ordered_Pointer);
    _mm_free(P_transpose_scaled_Block_col_ordered_Pointer);
    _mm_free(GQWeight_det_J_0xi_Pointer);
}

int main(int argc, char* argv[]) {
    
    ChMatrixNMc<double, 9, 3> e0_bar;
    e0_bar << 0, 0, 0,
        0, 1, 0,
        0, 0, 1,
        1, 0, 0,
        0, 1, 0,
        0, 0, 1,
        0.5, 0, 0,
        0, 1, 0,
        0, 0, 1;

    ChMatrixNMc<double, 9, 3> e_bar = e0_bar;
    ChMatrixNMc<double, 9, 3> edot_bar;
    edot_bar.setZero();

    std::cout << "Generalized Internal Force - Small Displacement, No Velocity = " << std::endl;
    Check_No_Damping_Internal_Force(e0_bar, e_bar);

    std::cout << std::endl << std::endl << "---------------------------" << std::endl << std::endl;
    std::cout << "Generalized Internal Force - Small Displacement, No Velocity = " << std::endl;
    e_bar(6, 2) = 0.001;
    Check_No_Damping_Internal_Force(e0_bar, e_bar);

    std::cout << std::endl << std::endl << "---------------------------" << std::endl << std::endl;
    std::cout << "Generalized Internal Force - No Displacement, No Velocity (With Damping Code) = " << std::endl;
    e_bar(6, 2) = 0.0;
    Check_Damping_Internal_Force(e0_bar, e_bar, edot_bar);

    std::cout << std::endl << std::endl << "---------------------------" << std::endl << std::endl;
    std::cout << "Generalized Internal Force - Small Displacement, No Velocity (With Damping Code) = " << std::endl;
    e_bar(6, 2) = 0.001;
    Check_Damping_Internal_Force(e0_bar, e_bar, edot_bar);

    std::cout << std::endl << std::endl << "---------------------------" << std::endl << std::endl;
    std::cout << "Generalized Internal Force - No Displacement, Small Velocity (With Damping Code) = " << std::endl;
    e_bar(6, 2) = 0.0;
    edot_bar(6, 2) = 0.001;
    Check_Damping_Internal_Force(e0_bar, e_bar, edot_bar);


    return(0);
}


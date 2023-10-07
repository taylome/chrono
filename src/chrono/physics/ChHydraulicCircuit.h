// =============================================================================
// PROJECT CHRONO - http://projectchrono.org
//
// Copyright (c) 2023 projectchrono.org
// All rights reserved.
//
// Use of this source code is governed by a BSD-style license that can be found
// in the LICENSE file at the top level of the distribution and at
// http://projectchrono.org/license-chrono.txt.
//
// =============================================================================
// Authors: Radu Serban
// =============================================================================
//
// Hydraulic circuit models based on the paper:
//   Rahikainen, J., Gonz�lez, F., Naya, M.�., Sopanen, J. and Mikkola, A.,
//   "On the cosimulation of multibody systems and hydraulic dynamics."
//   Multibody System Dynamics, 50(2), pp.143-167, 2020.
//
// and the lumped fluid approach from:
//   Watton, J., "Fluid Power Systems: Modeling, Simulation, Analog and
//   Microcomputer Control." Prentice-Hall, New York, 1989
// =============================================================================

#ifndef CH_HYDRAULIC_CIRCUIT_H
#define CH_HYDRAULIC_CIRCUIT_H

#include "chrono/physics/ChExternalDynamics.h"
#include "chrono/physics/ChBody.h"
#include "chrono/motion_functions/ChFunction.h"

namespace chrono {

typedef std::pair<double, double> double2;

/// ChHydraulicCylinder - a simple hydraulic cylinder
/// Schematic:
/// <pre>
///
///    ____________________
///   |  1  |  2           |        F+
///   |     |-------------------    -->
///   |_____|______________|        s+
///
///     l1         l2
///   <----><------------->
///           lTotal
///   <------------------->
///               s
///   <------------------------>
///
/// </pre>
class ChHydraulicCylinder {
  public:
    ChHydraulicCylinder();

    const double2& GetAreas() const { return A; }

    double2 ComputeChamberLengths(double Delta_s) const;

    double2 ComputeChamberVolumes(const double2& L) const;

    double EvalForce(const double2& p, double Delta_s, double sd);

  private:
    double pistonD = 0.08;  ///< piston diameter [m]
    double rodD = 0.035;    ///< piston rod diameter [m]
    double pistonL = 0.3;   ///< piston length [m]

    double2 L0 = {0.05, 0.25};  ///< initial lengths (piston-side and rod-side) [m]

    double2 A;  ///< areas (piston-side and rod-side) [m^2]

    bool length_exceeded;
};

/// ChHydraulicDirectionalValve4x3 - a computational model of 4 / 3 directional valve
/// Schematic:
/// <pre>
///
///                        p4 p3
///                ______ ______ ______
///               | |  ^ | T T | ^  / |
///            _ _| |  | |         X  |_ _
///  <-U+>    |_/_|_v__|_|_T_T_|_v__\_|_/_|
///                        p1 p2
///
/// </pre>
class ChHydraulicDirectionalValve4x3 {
  public:
    ChHydraulicDirectionalValve4x3();

    // Ud = Ud(t, U)
    double EvaluateSpoolPositionRate(double t, double U, double Uref);

    double2 ComputeVolumeFlows(double p1, double p2, double p3, double p4, double U);

  private:
    double linear_limit = 2e5;  ///< laminar flow rate limit of 2 bar [N/m^2]
    double dead_zone = 0.1e-5;  ///< limit for shut valve [m]
    double fm45 = 35.0;         ///< -45 degree phase shift frequency [Hz]
    double Cv;                  ///< semi-empirical flow rate coefficionet

    double time_constant;
};

/// ChHydraulicThrottleValve - a semi-empirical model of a throttle valve
/// Schematic:
/// <pre>
///
///     p2
///     |
///    )|( | Q
///     |
///     p1
///
/// </pre>
class ChHydraulicThrottleValve {
  public:
    ChHydraulicThrottleValve();

    double ComputeVolumeFlow(double p1, double p2);

  private:
    double Do = 850;  ///< oil density [kg/m^3]

    double Cd = 0.8;         ///< flow discharge coefficient of the orifice
    double valveD = 0.006;   ///< valve orifice diameter [m]
    double lin_limit = 2e5;  ///< laminar flow rate limit of 2 bar [N/m^2]
    double Cv;               ///< semi-empirical flow rate coefficient
};

}  // end namespace chrono

#endif

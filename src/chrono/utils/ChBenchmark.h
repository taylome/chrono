// =============================================================================
// PROJECT CHRONO - http://projectchrono.org
//
// Copyright (c) 2016 projectchrono.org
// All right reserved.
//
// Use of this source code is governed by a BSD-style license that can be found
// in the LICENSE file at the top level of the distribution and at
// http://projectchrono.org/license-chrono.txt.
//
// =============================================================================
// Authors: Radu Serban
// =============================================================================
//
// Utilities for performance benchmarking of Chrono simulations using the Google
// benchmark framework.
//
// =============================================================================

#ifndef CH_BENCHMARK_H
#define CH_BENCHMARK_H

#include "benchmark/benchmark.h"
#include "chrono/physics/ChSystem.h"
#include "chrono/solver/ChDirectSolverLS.h"

namespace chrono {
namespace utils {

/// Base class for a Chrono benchmark test.
/// A derived class should set up a complete Chrono model in its constructor and implement
/// GetSystem (to return a pointer to the underlying Chrono system) and ExecuteStep (to perform
/// all operations required to advance the system state by one time step).
/// Timing information for various phases of the simulation is collected for a sequence of steps.
class ChBenchmarkTest {
  public:
    ChBenchmarkTest();
    virtual ~ChBenchmarkTest() {}

    virtual void ExecuteStep() = 0;
    virtual ChSystem* GetSystem() = 0;

    void Simulate(int num_steps);
    void ResetTimers();

    double m_timer_step;                      ///< time for performing simulation
    double m_timer_advance;                   ///< time for integration
    double m_timer_jacobian;                  ///< time for evaluating/loading Jacobian data
    double m_timer_setup;                     ///< time for solver setup
    double m_timer_setup_assembly;            ///< time for solver setup assembly
    double m_timer_setup_solver;              ///< time for solver setup solver call
    double m_timer_solver;                    ///< time for solver solve
    double m_timer_solver_assembly;           ///< time for solver solve assembly
    double m_timer_solver_solver;             ///< time for solver solve solver call
    double m_timer_collision;                 ///< time for collision detection
    double m_timer_collision_broad;           ///< time for broad-phase collision
    double m_timer_collision_narrow;          ///< time for narrow-phase collision
    double m_timer_update;                    ///< time for system update
    double m_timer_fea_internal_frc;          ///< time for FEA internal force calculations
    double m_timer_fea_jacobian;              ///< time for FEA Jacobian calculations
    unsigned int m_counter_fea_internal_frc;  ///< counter for the number of times the FEA internal force function is
                                              ///< directly called
    unsigned int
        m_counter_KRM_load;  ///< counter for the number of times the FEA Jacobian evaluation is directly called
};

inline ChBenchmarkTest::ChBenchmarkTest()
    : m_timer_step(0),
      m_timer_advance(0),
      m_timer_jacobian(0),
      m_timer_setup(0),
      m_timer_setup_assembly(0),
      m_timer_setup_solver(0),
      m_timer_solver(0),
      m_timer_solver_assembly(0),
      m_timer_solver_solver(0),
      m_timer_collision(0),
      m_timer_collision_broad(0),
      m_timer_collision_narrow(0),
      m_timer_update(0),
      m_timer_fea_internal_frc(0),
      m_timer_fea_jacobian(0),
      m_counter_fea_internal_frc(0),
      m_counter_KRM_load(0) {}

inline void ChBenchmarkTest::Simulate(int num_steps) {
    ////std::cout << "  simulate from t=" << GetSystem()->GetChTime() << " for steps=" << num_steps << std::endl;
    ResetTimers();
    auto LS = std::dynamic_pointer_cast<ChDirectSolverLS>(GetSystem()->GetSolver());
    for (int i = 0; i < num_steps; i++) {
        auto MeshList = GetSystem()->Get_meshlist();
        for (auto& Mesh : MeshList) {
            Mesh->ResetTimers();
            Mesh->ResetCounters();
        }
        if (LS != NULL) {  // Direct Solver
            LS->ResetTimers();
        }

        ExecuteStep();
        m_timer_step += GetSystem()->GetTimerStep();
        m_timer_advance += GetSystem()->GetTimerAdvance();
        m_timer_jacobian += GetSystem()->GetTimerJacobian();
        m_timer_setup += GetSystem()->GetTimerSetup();
        m_timer_solver += GetSystem()->GetTimerSolver();
        m_timer_collision += GetSystem()->GetTimerCollision();
        m_timer_collision_broad += GetSystem()->GetTimerCollisionBroad();
        m_timer_collision_narrow += GetSystem()->GetTimerCollisionNarrow();
        m_timer_update += GetSystem()->GetTimerUpdate();

        if (LS != NULL) {  // Direct Solver
            m_timer_setup_assembly += LS->GetTimeSetup_Assembly();
            m_timer_setup_solver += LS->GetTimeSetup_SolverCall();
            m_timer_solver_assembly += LS->GetTimeSolve_Assembly();
            m_timer_solver_solver += LS->GetTimeSolve_SolverCall();
        }
        // Accumulate the internal force and Jacobian timers across all the FEA mesh containers
        // auto MeshList = GetSystem()->Get_meshlist();
        for (auto& Mesh : MeshList) {
            m_timer_fea_internal_frc += Mesh->GetTimeInternalForces();
            m_timer_fea_jacobian += Mesh->GetTimeJacobianLoad();
            m_counter_fea_internal_frc += Mesh->GetNumCallsInternalForces();
            m_counter_KRM_load += Mesh->GetNumCallsJacobianLoad();
        }
    }
}

inline void ChBenchmarkTest::ResetTimers() {
    m_timer_step = 0;
    m_timer_advance = 0;
    m_timer_jacobian = 0;
    m_timer_setup = 0;
    m_timer_setup_assembly = 0;
    m_timer_setup_solver = 0;
    m_timer_solver = 0;
    m_timer_solver_assembly = 0;
    m_timer_solver_solver = 0;
    m_timer_collision = 0;
    m_timer_collision_broad = 0;
    m_timer_collision_narrow = 0;
    m_timer_update = 0;
    m_timer_fea_internal_frc = 0;
    m_timer_fea_jacobian = 0;
    m_counter_fea_internal_frc = 0;
    m_counter_KRM_load = 0;
}

// =============================================================================

/// Define and register a test named TEST_NAME using the specified ChBenchmark TEST.
/// This method benchmarks consecutive (in time) simulation batches and is therefore
/// appropriate for cases where the cost per step is expected to be relatively uniform.
/// An initial SKIP_STEPS integration steps are performed for hot start, after which
/// measurements are conducted for batches of SIM_STEPS integration steps.
/// The test is repeated REPETITIONS number of times, to collect statistics.
/// Note that each reported benchmark result may require simulations of several batches
/// (controlled by the benchmark library in order to stabilize timing results).
#define CH_BM_SIMULATION_LOOP(TEST_NAME, TEST, SKIP_STEPS, SIM_STEPS, REPETITIONS) \
    using TEST_NAME = chrono::utils::ChBenchmarkFixture<TEST, SKIP_STEPS>;         \
    BENCHMARK_DEFINE_F(TEST_NAME, SimulateLoop)(benchmark::State & st) {           \
        while (st.KeepRunning()) {                                                 \
            m_test->Simulate(SIM_STEPS);                                           \
        }                                                                          \
        Report(st);                                                                \
    }                                                                              \
    BENCHMARK_REGISTER_F(TEST_NAME, SimulateLoop)->Unit(benchmark::kMillisecond)->Repetitions(REPETITIONS);

/// Define and register a test named TEST_NAME using the specified ChBenchmark TEST.
/// This method benchmarks a single simulation interval and is appropriate for cases
/// where the cost of simulating a given length time interval can vary significantly
/// from interval to interval.
/// For each measurement, the underlying model is recreated from scratch. An initial
/// SKIP_STEPS integration steps are performed for hot start, after which a single
/// batch of SIM_STEPS is timed and recorded.
/// The test is repeated REPETITIONS number of times, to collect statistics.
#define CH_BM_SIMULATION_ONCE(TEST_NAME, TEST, SKIP_STEPS, SIM_STEPS, REPETITIONS) \
    using TEST_NAME = chrono::utils::ChBenchmarkFixture<TEST, 0>;                  \
    BENCHMARK_DEFINE_F(TEST_NAME, SimulateOnce)(benchmark::State & st) {           \
        Reset(SKIP_STEPS);                                                         \
        while (st.KeepRunning()) {                                                 \
            m_test->Simulate(SIM_STEPS);                                           \
        }                                                                          \
        Report(st);                                                                \
    }                                                                              \
    BENCHMARK_REGISTER_F(TEST_NAME, SimulateOnce)                                  \
        ->Unit(benchmark::kMillisecond)                                            \
        ->Iterations(1)                                                            \
        ->Repetitions(REPETITIONS);

// =============================================================================

/// Generic benchmark fixture for Chrono tests.
/// The first template parameter is a ChBenchmarkTest.
/// The second template parameter is the initial number of simulation steps (hot start).
template <typename TEST, int SKIP>
class ChBenchmarkFixture : public ::benchmark::Fixture {
  public:
    ChBenchmarkFixture() : m_test(nullptr) {
        ////std::cout << "CREATE TEST" << std::endl;
        if (SKIP != 0) {
            m_test = new TEST();
            m_test->Simulate(SKIP);
        }
    }

    ~ChBenchmarkFixture() { delete m_test; }

    void Report(benchmark::State& st) {
        st.counters["Step_Total"] = m_test->m_timer_step * 1e3;
        st.counters["Step_Advance"] = m_test->m_timer_advance * 1e3;
        st.counters["Step_Update"] = m_test->m_timer_update * 1e3;
        st.counters["LS_Jacobian"] = m_test->m_timer_jacobian * 1e3;
        st.counters["LS_Setup"] = m_test->m_timer_setup * 1e3;
        st.counters["LS_Setup_Asm"] = m_test->m_timer_setup_assembly * 1e3;
        st.counters["LS_Setup_Solver"] = m_test->m_timer_setup_solver * 1e3;
        st.counters["LS_Solve"] = m_test->m_timer_solver * 1e3;
        st.counters["LS_Solve_Asm"] = m_test->m_timer_solver_assembly * 1e3;
        st.counters["LS_Solve_Solver"] = m_test->m_timer_solver_solver * 1e3;
        st.counters["CD_Total"] = m_test->m_timer_collision * 1e3;
        st.counters["CD_Broad"] = m_test->m_timer_collision_broad * 1e3;
        st.counters["CD_Narrow"] = m_test->m_timer_collision_narrow * 1e3;
        st.counters["FEA_InternalFrc"] = m_test->m_timer_fea_internal_frc * 1e3;
        st.counters["FEA_Jacobian"] = m_test->m_timer_fea_jacobian * 1e3;
        st.counters["FEA_InternalFrc_Calls"] = m_test->m_counter_fea_internal_frc;
        st.counters["FEA_Jacobian_Calls"] = m_test->m_counter_KRM_load;
    }

    void Reset(int num_init_steps) {
        ////std::cout << "RESET" << std::endl;
        delete m_test;
        m_test = new TEST();
        m_test->Simulate(num_init_steps);
    }

    TEST* m_test;
};

}  // end namespace utils
}  // end namespace chrono

#endif

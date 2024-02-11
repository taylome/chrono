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
// Authors: Radu Serban
// =============================================================================
//
// Ray intersection test
//
// =============================================================================

#include "chrono/physics/ChSystemSMC.h"
#include "chrono/physics/ChBodyEasy.h"
#include "chrono/core/ChMathematics.h"
#include "chrono/utils/ChUtilsSamplers.h"
#include "chrono/collision/multicore/ChCollisionSystemMulticore.h"

#include "chrono_irrlicht/ChVisualSystemIrrlicht.h"

using namespace chrono;
using namespace chrono::irrlicht;

using std::cout;
using std::endl;

// =============================================================================

// Collision detection system
ChCollisionSystem::Type collision_type = ChCollisionSystem::Type::MULTICORE;

// =============================================================================

class RayCaster {
  public:
    RayCaster(ChSystem* sys, const ChFrame<>& origin, const ChVector2d& dims, double spacing);

    const std::vector<ChVector3d>& GetPoints() const { return m_points; }

    void Update();

  private:
    ChSystem* m_sys;
    ChFrame<> m_origin;
    ChVector2d m_dims;
    double m_spacing;
    std::shared_ptr<ChBody> m_body;
    std::shared_ptr<ChGlyphs> m_glyphs;
    std::vector<ChVector3d> m_points;
};

RayCaster::RayCaster(ChSystem* sys, const ChFrame<>& origin, const ChVector2d& dims, double spacing)
    : m_sys(sys), m_origin(origin), m_dims(dims), m_spacing(spacing) {
    m_body = chrono_types::make_shared<ChBody>();
    m_body->SetBodyFixed(true);
    m_body->SetCollide(false);
    sys->AddBody(m_body);

    m_glyphs = chrono_types::make_shared<ChGlyphs>();
    m_glyphs->SetGlyphsSize(0.1);
    m_glyphs->SetZbufferHide(true);
    m_glyphs->SetDrawMode(ChGlyphs::GLYPH_POINT);
    m_body->AddVisualShape(m_glyphs);
}

void RayCaster::Update() {
    m_points.clear();

    ChVector3d dir = m_origin.GetA().GetAxisZ();
    int nx = static_cast<int>(std::round(m_dims.x() / m_spacing));
    int ny = static_cast<int>(std::round(m_dims.y() / m_spacing));
    for (int ix = 0; ix < nx; ix++) {
        for (int iy = 0; iy < ny; iy++) {
            double x_local = -0.5 * m_dims.x() + ix * m_spacing;
            double y_local = -0.5 * m_dims.y() + iy * m_spacing;
            ChVector3d from = m_origin.TransformPointLocalToParent(ChVector3d(x_local, y_local, 0.0));
            ChVector3d to = from + dir * 100;
            ChCollisionSystem::ChRayhitResult result;
            m_sys->GetCollisionSystem()->RayHit(from, to, result);
            if (result.hit)
                m_points.push_back(result.abs_hitPoint);
        }
    }

    m_glyphs->Reserve(0);
    for (unsigned int id = 0; id < m_points.size(); id++) {
        m_glyphs->SetGlyphPoint(id, m_points[id], ChColor(1, 1, 0));
    }
}

// =============================================================================

// Various collections of shapes.
// Attention!
// - we must enable collision for each shape (to be able to cast rays in the collision system)
// - however, we do not want any two shapes to generate contact forces, so we place all shapes in the same collision
//   family and disable collision with that family
// - currently, for the Chrono collision system, the collision family must be set *before* adding the collision model
//   to the collision system (i.e., before adding the body to the system)

void CreateSpheres(ChSystemSMC& sys) {
    auto mat = chrono_types::make_shared<ChContactMaterialSMC>();

    auto s1 = chrono_types::make_shared<ChBodyEasySphere>(2.0, 1, mat);
    s1->SetPos(ChVector3d(0, 0, 0));
    s1->GetVisualShape(0)->SetColor(ChColor(0.4f, 0, 0));
    s1->GetCollisionModel()->SetFamily(1);
    s1->GetCollisionModel()->SetFamilyMaskNoCollisionWithFamily(1);
    sys.Add(s1);

    auto s2 = chrono_types::make_shared<ChBodyEasySphere>(2.0, 1, mat);
    s2->SetPos(ChVector3d(2, 0, 3));
    s2->GetVisualShape(0)->SetColor(ChColor(0.4f, 0, 0));
    s2->GetCollisionModel()->SetFamily(1);
    s2->GetCollisionModel()->SetFamilyMaskNoCollisionWithFamily(1);
    sys.Add(s2);
}

void CreateBoxes(ChSystemSMC& sys) {
    auto mat = chrono_types::make_shared<ChContactMaterialSMC>();

    auto b1 = chrono_types::make_shared<ChBodyEasyBox>(3.0, 2.0, 1.0, 1, mat);
    b1->SetPos(ChVector3d(0, 0, 0));
    b1->SetRot(ChQuaternion<>(ChRandom(), ChRandom(), ChRandom(), ChRandom()).GetNormalized());
    b1->GetVisualShape(0)->SetColor(ChColor(0, 0.4f, 0));
    b1->GetCollisionModel()->SetFamily(1);
    b1->GetCollisionModel()->SetFamilyMaskNoCollisionWithFamily(1);
    sys.Add(b1);

    auto b2 = chrono_types::make_shared<ChBodyEasyBox>(5.0, 4.0, 1.0, 1, mat);
    b2->SetPos(ChVector3d(0, 0, +3));
    b2->GetVisualShape(0)->SetColor(ChColor(0, 0.4f, 0));
    b2->GetCollisionModel()->SetFamily(1);
    b2->GetCollisionModel()->SetFamilyMaskNoCollisionWithFamily(1);
    sys.Add(b2);
}

void CreateCylinders(ChSystemSMC& sys) {
    auto mat = chrono_types::make_shared<ChContactMaterialSMC>();

    auto c1 = chrono_types::make_shared<ChBodyEasyCylinder>(geometry::ChAxis::Y, 1.0, 2.0, 1, mat);
    c1->SetPos(ChVector3d(0, 0, 0));
    c1->SetRot(ChQuaternion<>(ChRandom(), ChRandom(), ChRandom(), ChRandom()).GetNormalized());
    c1->GetVisualShape(0)->SetColor(ChColor(0, 0, 0.4f));
    c1->GetCollisionModel()->SetFamily(1);
    c1->GetCollisionModel()->SetFamilyMaskNoCollisionWithFamily(1);
    sys.Add(c1);

    auto c2 = chrono_types::make_shared<ChBodyEasyCylinder>(geometry::ChAxis::Y, 2.0, 4.0, 1, mat);
    c2->SetPos(ChVector3d(0, 0, 3));
    c2->SetRot(Q_from_AngZ(CH_C_PI / 4));
    c2->GetVisualShape(0)->SetColor(ChColor(0.6f, 0.6f, 0.7f));
    c2->GetCollisionModel()->SetFamily(1);
    c2->GetCollisionModel()->SetFamilyMaskNoCollisionWithFamily(1);
    sys.Add(c2);
}

void CreateShapes(ChSystemSMC& sys) {
    // Create multiple bodies and collision shapes
    auto mat = chrono_types::make_shared<ChContactMaterialSMC>();

    double scale = 2.0;
    utils::PDSampler<> sampler(2 * scale);
    auto points = sampler.SampleBox(ChVector3d(0, 0, 0), ChVector3d(10, 10, 10));

    for (int i = 0; i < points.size() / 3; i++) {
        auto s = chrono_types::make_shared<ChBodyEasySphere>(0.75 * scale, 1, mat);
        s->SetPos(points[3 * i + 0]);
        s->GetVisualShape(0)->SetColor(ChColor(0.4f, 0, 0));
        s->GetCollisionModel()->SetFamily(1);
        s->GetCollisionModel()->SetFamilyMaskNoCollisionWithFamily(1);
        sys.Add(s);

        auto b =
            chrono_types::make_shared<ChBodyEasyBox>(1.0 * scale, 1.5 * scale, 1.25 * scale, 1, mat);
        b->SetPos(points[3 * i + 1]);
        b->SetRot(ChQuaternion<>(ChRandom(), ChRandom(), ChRandom(), ChRandom()).GetNormalized());
        b->GetVisualShape(0)->SetColor(ChColor(0, 0.4f, 0));
        b->GetCollisionModel()->SetFamily(1);
        b->GetCollisionModel()->SetFamilyMaskNoCollisionWithFamily(1);
        sys.Add(b);

        auto c = chrono_types::make_shared<ChBodyEasyCylinder>(geometry::ChAxis::Y, 0.75 * scale, 0.75 * scale, 1, mat);
        c->SetPos(points[3 * i + 2]);
        c->SetRot(ChQuaternion<>(ChRandom(), ChRandom(), ChRandom(), ChRandom()).GetNormalized());
        c->GetVisualShape(0)->SetColor(ChColor(0, 0, 0.4f));
        c->GetCollisionModel()->SetFamily(1);
        c->GetCollisionModel()->SetFamilyMaskNoCollisionWithFamily(1);
        sys.Add(c);
    }
}

void CreateMeshes(ChSystemSMC& sys) {
    auto mat = chrono_types::make_shared<ChContactMaterialSMC>();

    auto trimesh = geometry::ChTriangleMeshConnected::CreateFromWavefrontFile(GetChronoDataFile("models/sphere.obj"));
    trimesh->Transform(ChVector3d(0), ChMatrix33<>(2));
    std::shared_ptr<ChVisualShapeTriangleMesh> vismesh(new ChVisualShapeTriangleMesh);
    vismesh->SetMesh(trimesh);
    vismesh->SetColor(ChColor(0.4f, 0, 0));

    auto m1 = chrono_types::make_shared<ChBody>();
    m1->AddVisualShape(vismesh);
    auto m1_shape = chrono_types::make_shared<ChCollisionShapeTriangleMesh>(mat, trimesh, false, false, 0.01);
    m1->AddCollisionShape(m1_shape);
    m1->SetCollide(true);
    sys.Add(m1);
}

// Create a set of boxes for testing broadphase ray intersection.
// Should be used with a (4x3x1) grid.
void CreateTestSet(ChSystemSMC& sys) {
    auto mat = chrono_types::make_shared<ChContactMaterialSMC>();

    /*
    std::vector<ChVector3d> loc = {
        ChVector3d(1, 1, 0),   //
        ChVector3d(8, 3, 0),   //
        ChVector3d(5, 5, 0),   //
        ChVector3d(13, 5, 0),  //
        ChVector3d(2, 7, 0),   //
        ChVector3d(3, 8, 0),   //
        ChVector3d(13, 8, 0),  //
        ChVector3d(19, 14, 0)  //
    };
    */

    std::vector<ChVector3d> loc = {
        ChVector3d(19, 1, 0),  //
        ChVector3d(8, 3, 0),   //
        ChVector3d(7, 7, 0),   //
        ChVector3d(13, 5, 0),  //
        ChVector3d(2, 7, 0),   //
        ChVector3d(3, 8, 0),   //
        ChVector3d(13, 8, 0),  //
        ChVector3d(1, 14, 0)   //
    };

    for (int i = 0; i < 8; i++) {
        auto b = chrono_types::make_shared<ChBodyEasyBox>(2.0, 2.0, 2.0, 1, mat);
        b->SetPos(loc[i] - ChVector3d(5, 5, 0));
        b->GetVisualShape(0)->SetColor(ChColor(0, 0.4f, 0));
        b->GetCollisionModel()->SetFamily(1);
        b->GetCollisionModel()->SetFamilyMaskNoCollisionWithFamily(1);
        sys.Add(b);
    }
}

// =============================================================================

int main(int argc, char* argv[]) {
    std::cout << "Copyright (c) 2017 projectchrono.org\nChrono version: " << CHRONO_VERSION << std::endl;

    bool rotate_shapes = true;
    bool draw_rays = true;

    // Create the system
    ChSystemSMC sys;
    sys.Set_G_acc(ChVector3d(0, 0, 0));
    sys.SetCollisionSystemType(collision_type);
    if (collision_type == ChCollisionSystem::Type::MULTICORE) {
        auto cd_chrono = std::static_pointer_cast<ChCollisionSystemMulticore>(sys.GetCollisionSystem());
        cd_chrono->SetBroadphaseGridResolution(ChVector3i(3, 3, 3));
        ////cd_chrono->SetBroadphaseGridResolution(ChVector3i(4, 3, 1));
    }

    ////CreateSpheres(sys);
    ////CreateBoxes(sys);
    ////CreateCylinders(sys);
    ////CreateMeshes(sys);
    ////CreateTestSet(sys);
    CreateShapes(sys);

    // Cast rays in collision models (in Z direction of specified frame)
    RayCaster caster(&sys, ChFrame<>(ChVector3d(0, 0, -20), Q_from_AngX(0)), ChVector2d(10, 10), 0.5);

    // Create the Irrlicht visualization system
    auto vis = chrono_types::make_shared<ChVisualSystemIrrlicht>();
    vis->AttachSystem(&sys);
    vis->SetWindowSize(800, 600);
    vis->SetWindowTitle("Ray intersection test");
    vis->Initialize();
    vis->AddLogo();
    vis->AddSkyBox();
    vis->AddCamera(ChVector3d(0, 0, -60));
    vis->AddTypicalLights();

    vis->GetActiveCamera()->setFOV(irr::core::PI / 10.0f);

    if (rotate_shapes) {
        for (auto& b : sys.Get_bodylist())
            b->SetWvel_loc(ChVector3d(0.1, 0.1, 0.1));
    }

    while (vis->Run()) {
        sys.DoStepDynamics(0.01);
        caster.Update();

        vis->BeginScene();
        vis->Render();

        if (draw_rays) {
            for (auto& p : caster.GetPoints()) {
                tools::drawSegment(vis.get(), p - ChVector3d(0, 0, 100), p, ChColor(0.5f, 0.5f, 0.5f), true);
            }
        }

        vis->EndScene();
    }

    return 0;
}

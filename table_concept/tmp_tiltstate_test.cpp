#include "src/table_belief.h"
#include <cstdio>
#include <random>
using namespace rc;
int main(){
  const TableBeliefState t0{0,0,0.74f,1.5f,1.0f,0.0f};
  const Eigen::Vector3f cam_top(0,0,0.74f+3);
  // camera-frame azimuth referenced to the TRUE optical axis (look-at true center); injection AND estimator share it
  auto make_frame=[&](Eigen::Vector3f cam,float t,std::mt19937&rng){
    std::uniform_real_distribution<float> U(-1,1); std::normal_distribution<float> nz(0,0.01f);
    Eigen::Vector2f ax(t0.cx-cam.x(),t0.cy-cam.y()); ax.normalize();
    TableFrame fr;
    for(int i=0;i<1500;i++){Eigen::Vector3f w(U(rng)*0.5f*t0.w,U(rng)*0.5f*t0.h,t0.H+nz(rng));
      Eigen::Vector3f d=w-cam;float z=d.norm();Eigen::Vector3f rh=d/z;Eigen::Vector2f rxy(d.x(),d.y());rxy.normalize();
      float az=std::atan2(ax.x()*rxy.y()-ax.y()*rxy.x(),ax.dot(rxy));
      fr.points.push_back(cam+(z+t*az)*rh); fr.point_azim.push_back(az);}
    fr.R.assign(fr.points.size(),0.03f*0.03f); fr.cam_origin=cam; fr.has_rays=true; fr.chain_cov_yaw=0.0f;
    return fr;
  };
  auto run=[&](float t_true,bool tilt_state,bool orbit){
    std::mt19937 rng(3131);
    TableBeliefState::use_quotient=true;
    TableBeliefParams pp; pp.footprint_residual=true; pp.footprint_moment_precision=0; pp.depth_tilt_std= tilt_state?0.05f:1e-7f;
    TableBelief b(t0,pp);
    for(int i=0;i<40;i++){auto fr=make_frame(cam_top,0.0f,rng);b.update(fr);}     // burn-in pins yaw
    for(int i=0;i<200;i++){float ang=0.44f+(orbit?0.5f*std::sin(i*0.15f):0.0f);
      Eigen::Vector3f cg(-3*std::cos(ang),-3*std::sin(ang),0.74f); auto fr=make_frame(cg,t_true,rng);b.update(fr);}
    float yaw=std::abs(std::remainder(b.state().yaw,(float)M_PI));
    TableBeliefState::use_quotient=false;
    return std::make_pair(yaw,b.state().t);
  };
  std::printf("=== tilt STATE ON, camera-frame azimuth + ORBIT (viewpoint diversity) ===\n");
  for(float t:{0.02f,0.08f,0.15f,0.30f}){auto r=run(t,true,true); std::printf("  t_true=%.2f: yaw %.3f rad (%.1f deg)  t_hat=%.4f\n",t,r.first,r.first*57.3f,r.second);}
  std::printf("=== control: tilt STATE OFF (should drift) ===\n");
  for(float t:{0.02f,0.15f}){auto r=run(t,false,true); std::printf("  t_true=%.2f: yaw %.1f deg\n",t,r.first*57.3f);}
  return 0;
}

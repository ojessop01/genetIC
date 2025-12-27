#ifndef _COSMOLOGY_ISOCURVATURE_HPP_INCLUDED
#define _COSMOLOGY_ISOCURVATURE_HPP_INCLUDED

namespace cosmology {

  // Isocurvature baryon-CDM correlation coefficient used for Grafic output.
  //
  // The accessor returns a reference to a function-local static so the CAMB
  // loader can compute the coefficient once (from z=0 transfer functions) and
  // Grafic can later read the same value when splitting baryons and CDM. The
  // initial seed (-0.0049) is overwritten as soon as CAMB computes the real
  // coefficient.
  inline double &isocurvature_alpha() {
    static double alpha = -0.0049;
    return alpha;
  }

  // Variance of the baryon-CDM relative velocity field (vb-vc) computed from
  // the CAMB transfer-function table.
  inline double &vbvc_variance() {
    static double variance = 0.0;
    return variance;
  }

} // namespace cosmology

#endif

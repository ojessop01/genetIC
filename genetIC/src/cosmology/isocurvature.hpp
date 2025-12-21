#ifndef _COSMOLOGY_ISOCURVATURE_HPP_INCLUDED
#define _COSMOLOGY_ISOCURVATURE_HPP_INCLUDED

namespace cosmology {

  // Redshift at which the isocurvature mode is defined
  //
  // Inline variables have a single shared definition across translation units, so
  // storing the configured target redshift here makes the value visible to both
  // the CAMB loader (which reads it to compute alpha) and the Grafic writer
  // (which needs it only for logging and context).
  inline double isocurvature_redshift = 99.0;

  // Isocurvature baryon-CDM correlation coefficient used for Grafic output.
  //
  // The accessor returns a reference to a function-local static. This mirrors
  // the inline variable behaviour above: every translation unit sees the same
  // underlying storage, so the value CAMB writes during construction is exactly
  // what Grafic later reads when splitting baryons and CDM. The initial seed
  // (-0.0049) is overwritten as soon as CAMB computes the real coefficient at
  // the configured target redshift.
  inline double &isocurvature_alpha() {
    static double alpha = -0.0049;
    return alpha;
  }

} // namespace cosmology

#endif

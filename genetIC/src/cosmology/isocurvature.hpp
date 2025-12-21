#ifndef _COSMOLOGY_ISOCURVATURE_HPP_INCLUDED
#define _COSMOLOGY_ISOCURVATURE_HPP_INCLUDED

namespace cosmology {

  // Redshift at which the isocurvature mode is defined
  inline double isocurvature_redshift = 99.0;

  // Isocurvature baryon-CDM correlation coefficient used for Grafic output
  inline double &isocurvature_alpha() {
    static double alpha = -0.0049;
    return alpha;
  }

} // namespace cosmology

#endif

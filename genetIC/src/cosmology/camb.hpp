#ifndef _CAMB_HPP_INCLUDED
#define _CAMB_HPP_INCLUDED

#include <cmath>
#include <complex>
#include <limits>
#include <map>
#include <memory>
#include <utility>
#include <vector>

#include "src/cosmology/parameters.hpp"
#include "src/io/input.hpp"
#include "src/simulation/particles/particle.hpp"
#include "src/tools/logging.hpp"
#include "src/tools/lru_cache.hpp"
#include "src/tools/numerics/interpolation.hpp"

/*!
    \namespace cosmology
    \brief Describe the cosmology adopted (parameters and transfer function)

    Allow to use cosmological parameters and to draw random fields with a cosmological power spectrum.
    The input transfer function is tied to CAMB format.
 */
namespace cosmology {

  /* \class CacheKeyComparator
   * Comparison class for pair<weak_ptr<...>,...>, using owner_less comparison on the weak_ptr
   * This enables maps with weak pointers as keys, as used by caching in the PowerSpectrum class.
   */
  template<typename F>
  struct CacheKeyComparator {
    bool operator()(const F &a, const F &b) const {
      return std::owner_less<typename F::first_type>()(a.first, b.first) ||
             (!std::owner_less<typename F::first_type>()(b.first, a.first) && a.second < b.second);
    }
  };

  size_t lru_cache_size = 10;

  /*! \class PowerSpectrum
   * \brief Abstract base class for power spectrum calculations.
   *
   * For practical runs, the CAMB child class performs the key work. For testing runs, a
   * PowerLawPowerSpectrum implementation is also provided.
   */
  template<typename DataType>
  class PowerSpectrum {
  public:
    using CoordinateType = tools::datatypes::strip_complex<DataType>;
    using FieldType = fields::Field<DataType, CoordinateType>;

  protected:
    using CacheKeyType =
      std::pair<std::weak_ptr<const grids::Grid<CoordinateType>>, particle::species>;

    mutable tools::lru_cache<CacheKeyType,
      std::shared_ptr<FieldType>,
      CacheKeyComparator<CacheKeyType>> calculatedCovariancesCache{lru_cache_size};

  public:
    virtual ~PowerSpectrum() = default;

    //! \brief Evaluate power spectrum for a given species at wavenumber k (Mpc/h), including the normalisation
    //! transferType specifies the transfer function to use (currently dm or baryon)
    virtual CoordinateType operator()(CoordinateType k, particle::species transferType) const = 0;

    //! Get the theoretical power spectrum appropriate for a given grid. This may be a cached copy if previously calculated.
    std::shared_ptr<fields::Field<DataType, CoordinateType>>
    getPowerSpectrumForGrid(const std::shared_ptr<const grids::Grid<CoordinateType>> &grid,
                            particle::species transferType = particle::species::dm) const {

      if (transferType == particle::species::whitenoise) {
        return nullptr;
      }

      if (lru_cache_size == 0) {
        return getPowerSpectrumForGridUncached(grid, transferType);
      }

      auto cacheKey =
        std::make_pair(std::weak_ptr<const grids::Grid<CoordinateType>>(grid), transferType);

      auto result = this->calculatedCovariancesCache.get(cacheKey);

      if (result == boost::none) {
        auto psForGrid = getPowerSpectrumForGridUncached(grid, transferType);
        this->calculatedCovariancesCache.insert(cacheKey, psForGrid);
        return psForGrid;
      } else {
        return result.get();
      }
    }

  protected:
    //! Calculate the theoretical power spectrum for a given grid
    virtual std::shared_ptr<fields::Field<DataType, CoordinateType>>
    getPowerSpectrumForGridUncached(std::shared_ptr<const grids::Grid<CoordinateType>> grid,
                                    particle::species transferType = particle::species::dm) const {

      CoordinateType norm = this->getPowerSpectrumNormalizationForGrid(*grid);

      auto P = std::make_shared<fields::Field<DataType, CoordinateType>>(*grid, true);

      P->forEachFourierCell(
        [norm, this, transferType](std::complex<CoordinateType>, CoordinateType kx,
                                  CoordinateType ky, CoordinateType kz) {
          CoordinateType k = std::sqrt(kx * kx + ky * ky + kz * kz);
          auto spec = std::complex<CoordinateType>((*this)(k, transferType) * norm, 0);
          return spec;
        });

      return P;
    }

  public:
    //! Return the box- and fft-dependent part of the normalisation of the power spectrum
    static CoordinateType
    getPowerSpectrumNormalizationForGrid(const grids::Grid<CoordinateType> &grid) {

      CoordinateType kw = 2. * M_PI / grid.thisGridSize;
      CoordinateType norm =
        kw * kw * kw / std::pow(static_cast<CoordinateType>(2) * M_PI,
                               static_cast<CoordinateType>(3)); // 1/V_box

      // FFT normalisation factor
      CoordinateType fft_normalisation = grid.size3;

      return norm * fft_normalisation;
    }
  };

  /*! \class PowerLawPowerSpectrum
   * \brief Pure power law power spectrum, for testing purposes only.
   *
   * ns is interpreted as the power law index, and sigma8 is ignored
   */
  template<typename DataType>
  class PowerLawPowerSpectrum : public PowerSpectrum<DataType> {
  protected:
    using typename PowerSpectrum<DataType>::CoordinateType;
    CoordinateType ns;
    CoordinateType amplitude;

  public:
    PowerLawPowerSpectrum(const CosmologicalParameters<CoordinateType> &cosmology,
                          CoordinateType amplitude)
      : ns(cosmology.ns), amplitude(amplitude) {}

    CoordinateType operator()(CoordinateType k, particle::species) const override {
      if (k == 0) {
        return 0;
      }
      return amplitude * std::pow(k, ns);
    }
  };

  /*! \class CAMB
   * \brief Provides power spectra by using transfer functions from CAMB output
   */
  template<typename DataType>
  class CAMB : public PowerSpectrum<DataType> {
    using typename PowerSpectrum<DataType>::CoordinateType;

  protected:
    // Store the cosmology so redshift (and any other fields) are accessible everywhere.
    CosmologicalParameters<CoordinateType> cosmology_;

    std::vector<double> kInterpolationPoints; //!< Wavenumbers read from CAMB file
    std::map<particle::species, std::vector<double>>
      speciesToInterpolationPoints; //!< Vector to store transfer functions
    std::vector<double>
      vbvcInterpolationPoints; //!< Transfer function for baryon-CDM relative velocity (vb-vc)

    const std::map<particle::species, size_t> speciesToCambColumn{
      {particle::species::dm, 1},
      {particle::species::baryon, 2},
      {particle::species::all, 6} // if using a single transfer function, use the column for total
    };
    //!< Columns of CAMB that we request for DM and baryons respectively

    std::map<particle::species, tools::numerics::LogInterpolator<double>>
      speciesToTransferFunction; //!< Interpolation functions

    CoordinateType amplitude;       //!< Amplitude of the initial power spectrum
    CoordinateType linearAmplitude; //!< Amplitude of the initial power spectrum
    CoordinateType ns;              //!< scalar spectral index of the initial power spectrum

    mutable CoordinateType kcamb_max_in_file; //!< Maximum CAMB wavenumber

  public:
    //! Import data from CAMB file and initialise the interpolation functions used to compute the transfer functions:
    CAMB(const CosmologicalParameters<CoordinateType> &cosmology, const std::string &filename,
         bool computeIsocurvature, bool computeVbvcVariance)
      : cosmology_(cosmology) {

      readLinesFromCambOutput(filename);

      for (auto i = speciesToInterpolationPoints.begin();
           i != speciesToInterpolationPoints.end(); ++i) {
        this->speciesToTransferFunction[i->first].initialise(kInterpolationPoints, i->second);
      }

      ns = cosmology_.ns;
      calculateOverallNormalization(cosmology_);

      if (computeIsocurvature) {
        const CoordinateType alpha = calculateAlphaCoefficientDiscrete();
        isocurvature_alpha() = static_cast<double>(alpha);
      } else {
        isocurvature_alpha() = 0.0;
      }

      if (computeVbvcVariance) {
        if (!vbvcInterpolationPoints.empty()) {
          const CoordinateType vbvc = calculateVbVcVarianceDiscrete();
          vbvc_variance() = static_cast<double>(vbvc);
        }
      } else {
        vbvc_variance() = 0.0;
      }
    }

    void computeIsocurvatureAlpha() const {
      const CoordinateType alpha = calculateAlphaCoefficientDiscrete();
      isocurvature_alpha() = static_cast<double>(alpha);
    }

    void computeVbvcVariance() const {
      if (vbvcInterpolationPoints.empty()) {
        throw std::runtime_error(
          "Cannot compute vb-vc variance: required CAMB transfer column is missing. Ensure you are using post 2015 CAMB transfer function table");
      }
      const CoordinateType vbvc = calculateVbVcVarianceDiscrete();
      vbvc_variance() = static_cast<double>(vbvc);
    }

    CoordinateType operator()(CoordinateType k, particle::species transferType) const override {
      CoordinateType linearTransfer = 0.0;

      if (k != 0) {
        linearTransfer = speciesToTransferFunction.at(transferType)(k);
      }

      if (k > kcamb_max_in_file) {
        kcamb_max_in_file = std::numeric_limits<CoordinateType>::max();
      }

      return amplitude * std::pow(k, ns) * linearTransfer * linearTransfer;
    }

    /*!
     * \brief Compute the baryon–CDM modulation coefficient
     *        \f$\alpha = \langle \delta_{bc}\,\delta_m\rangle /
     *                    \langle \delta_m^2 \rangle \f$
     *        using the discrete CAMB transfer-function table.
     *
     * Physically, α measures the correlation between the baryon–CDM isocurvature
     * mode and the total matter density
     */
    CoordinateType calculateAlphaCoefficientDiscrete() const {
    
      // Retrieve CAMB transfer columns
      auto it_c = speciesToInterpolationPoints.find(particle::species::dm);
      auto it_b = speciesToInterpolationPoints.find(particle::species::baryon);
      auto it_m = speciesToInterpolationPoints.find(particle::species::all);
    
      // Defensive: ensure all required species are present
      if (it_c == speciesToInterpolationPoints.end() ||
          it_b == speciesToInterpolationPoints.end() ||
          it_m == speciesToInterpolationPoints.end()) {
        throw std::runtime_error(
          "Cannot compute alpha: required CAMB transfer columns (dm, baryon, all) are missing.");
      }
    
      const auto &kcamb = kInterpolationPoints;
      const auto &Tc = it_c->second;
      const auto &Tb = it_b->second;
      const auto &Tm = it_m->second;
    
      // Determine safe usable table length
      size_t n = kcamb.size();
      if (Tc.size() < n) n = Tc.size();
      if (Tb.size() < n) n = Tb.size();
      if (Tm.size() < n) n = Tm.size();
    
      if (n < 2) {
        throw std::runtime_error(
          "Cannot compute alpha: CAMB transfer table has insufficient points.");
      }
    
      CoordinateType num = 0;  // numerator integral <δ_bc δ_m>
      CoordinateType den = 0;  // denominator integral <δ_m^2>
    
      for (size_t i = 0; i + 1 < n; ++i) {
    
        const CoordinateType k1 = static_cast<CoordinateType>(kcamb[i]);
        const CoordinateType k2 = static_cast<CoordinateType>(kcamb[i + 1]);
    
        // Physically meaningful only for positive k
        if (k1 <= 0 || k2 <= 0) continue;
    
        // Logarithmic bin width in k
        const CoordinateType dlnk = std::log(k2 / k1);
    
        // Baryon–CDM mode at each endpoint
        const CoordinateType Tbc1 = static_cast<CoordinateType>(Tb[i] - Tc[i]);
        const CoordinateType Tbc2 = static_cast<CoordinateType>(Tb[i + 1] - Tc[i + 1]);
    
        // Total matter mode
        const CoordinateType Tm1 = static_cast<CoordinateType>(Tm[i]);
        const CoordinateType Tm2 = static_cast<CoordinateType>(Tm[i + 1]);
    
        // Cross spectra
        const CoordinateType Pmbc1 = std::pow(k1, ns + 3) * Tbc1 * Tm1;
        const CoordinateType Pmbc2 = std::pow(k2, ns + 3) * Tbc2 * Tm2;
    
        const CoordinateType Pm1 = std::pow(k1, ns + 3) * Tm1 * Tm1;
        const CoordinateType Pm2 = std::pow(k2, ns + 3) * Tm2 * Tm2;
    
        // Trapezoidal accumulation in log–k
        num += 0.5f * (Pmbc1 + Pmbc2) * dlnk;
        den += 0.5f * (Pm1   + Pm2)   * dlnk;
      }
    
      // Denominator must be strictly non-zero for meaningful alpha
      if (den == 0.0f) {
        throw std::runtime_error(
          "Cannot compute alpha: denominator integral is zero.");
      }
    
      const CoordinateType alpha = num / den;

      // ------------------------------------------------------------------
      // Physical sanity checks for ΛCDM alpha.
      // Expectation:
      //   alpha < 0  (anti-correlation of δ_bc with δ_m)
      //   |alpha| < 1  (bounded correlation strength)
      // ------------------------------------------------------------------
      if (!std::isfinite(alpha)) {
        throw std::runtime_error("Cannot compute alpha: result is NaN or infinite.");
      }
      
      // Small tolerance to allow rounding noise
      constexpr CoordinateType eps = static_cast<CoordinateType>(1e-6);
      
      // Must be negative
      if (alpha >= -eps) {
        throw std::runtime_error(
          "Computed alpha violates ΛCDM expectation: alpha must be negative.");
      }
      
      // Must have magnitude < 1
      if (std::abs(alpha) >= static_cast<CoordinateType>(1.0) + eps) {
        throw std::runtime_error(
          "Computed alpha violates ΛCDM expectation: |alpha| must be < 1.");
      }
    
      logging::entry()
        << "Computed isocurvature alpha coefficient = " << alpha
        << std::endl;
    
      return alpha;
    }
      /*!
     * \brief Compute the RMS baryon–CDM streaming velocity σ_{v_bc}(z)
     *
     * This routine computes the variance of the baryon–CDM relative velocity
     * field using the CAMB-provided transfer function for the vb–vc mode,
     * sampled discretely in k–space. The integral is evaluated in log–k using
     * a trapezoidal rule.
     */
    CoordinateType calculateVbVcVarianceDiscrete() const {
    
      // Ensure the transfer function exists
      if (vbvcInterpolationPoints.empty()) {
        throw std::runtime_error(
          "Cannot compute vb-vc variance: required CAMB transfer column is missing.");
      }
    
      // Speed of light in km/s (used to convert to physical velocities)
      constexpr CoordinateType C_KMS = static_cast<CoordinateType>(299792.458);
    
      const auto &kcamb = kInterpolationPoints;
      const auto &Tvbvc = vbvcInterpolationPoints;
    
      // Number of usable samples
      size_t n = std::min(kcamb.size(), Tvbvc.size());
      if (n < 2) {
        throw std::runtime_error(
          "Cannot compute vb-vc variance: CAMB transfer table has insufficient points.");
      }
    
      CoordinateType variance = 0;
    
      // Integrate in log–k using trapezoidal rule.
      // We integrate Δ²(k) = linearAmplitude k^{n_s+3} T^2 / (2π²).
      for (size_t i = 0; i + 1 < n; ++i) {
    
        const CoordinateType k1 = static_cast<CoordinateType>(kcamb[i]);
        const CoordinateType k2 = static_cast<CoordinateType>(kcamb[i + 1]);
        if (k1 <= 0 || k2 <= 0) continue;
    
        // Logarithmic spacing
        const CoordinateType dlnk = std::log(k2 / k1);
    
        const CoordinateType T1 = static_cast<CoordinateType>(Tvbvc[i]);
        const CoordinateType T2 = static_cast<CoordinateType>(Tvbvc[i + 1]);
    
        // Power spectrum contributions at k1 and k2 (CAMB convention)
        const CoordinateType P1 =
          linearAmplitude * std::pow(k1, ns + 3) * T1 * T1 / (2 * M_PI * M_PI);
    
        const CoordinateType P2 =
          linearAmplitude * std::pow(k2, ns + 3) * T2 * T2 / (2 * M_PI * M_PI);
    
        // Trapezoidal step in ln k
        variance += static_cast<CoordinateType>(0.5) * (P1 + P2) * dlnk;
      }
    
      // Convert variance → RMS streaming velocity.
      //
      // Multiply by c to convert dimensionless transfer → km/s,
      // and multiply by (1+z) for the linear redshift evolution of v_bc.
      const CoordinateType sigma =
        (variance > 0)
          ? std::sqrt(variance) * C_KMS * (static_cast<CoordinateType>(1) + cosmology_.redshift)
          : static_cast<CoordinateType>(0);
    
      // Latex-friendly logging output
      logging::entry()
        << "Computed v_bc RMS:"
        << "(z=)" << cosmology_.redshift
        << "=" << sigma
        << "km/s"
        << std::endl;
    
      return sigma;
    }

  protected:
    //! \brief Import data from CAMB file.
    //! Both pre-2015 and post-2015 formats can be used, and the function will detect which.
    void readLinesFromCambOutput(std::string filename) {
      kInterpolationPoints.clear();
      speciesToInterpolationPoints.clear();
      vbvcInterpolationPoints.clear();

      const int c_old_camb = 7;   // number of columns pre 2015
      const int c_new_camb = 13;  // number of columns post 2015
      const int vbvc_column_index = 12; // 0-based index for the 13th column
      int numCols;
      size_t j;

      std::vector<double> input;
      io::getBufferIgnoringColumnHeaders(input, filename);

      numCols = io::getNumberOfColumns(filename);

      if (numCols == c_old_camb) {
#ifdef DEBUG_INFO
        logging::entry() << "Using pre 2015 CAMB transfer function" << std::endl;
#endif
      } else if (numCols == c_new_camb) {
#ifdef DEBUG_INFO
        logging::entry() << "Using post 2015 CAMB transfer function" << std::endl;
#endif
      } else {
        throw std::runtime_error("CAMB transfer file doesn't have a sensible number of columns");
      }

      CoordinateType transferNormalisation = static_cast<CoordinateType>(input[1]);

      for (j = 0; j < input.size() / static_cast<size_t>(numCols); j++) {
        if (input[numCols * j] > 0) {
          kInterpolationPoints.push_back(static_cast<CoordinateType>(input[numCols * j]));

          for (auto i = speciesToCambColumn.begin(); i != speciesToCambColumn.end(); ++i) {
            speciesToInterpolationPoints[i->first].push_back(
              static_cast<CoordinateType>(input[numCols * j + i->second]) / transferNormalisation);
          }

          if (numCols == c_new_camb) {
            vbvcInterpolationPoints.push_back(
              static_cast<CoordinateType>(input[numCols * j + vbvc_column_index]) / transferNormalisation);
          }
        }
      }

      kcamb_max_in_file = static_cast<CoordinateType>(kInterpolationPoints.back());
    }

    //! Calculate the theoretical power spectrum for a given grid
    std::shared_ptr<fields::Field<DataType, CoordinateType>>
    getPowerSpectrumForGridUncached(std::shared_ptr<const grids::Grid<CoordinateType>> grid,
                                    particle::species transferType = particle::species::dm) const override {

      auto P = PowerSpectrum<DataType>::getPowerSpectrumForGridUncached(grid, transferType);

      if (kcamb_max_in_file == std::numeric_limits<CoordinateType>::max()) {
        logging::entry(logging::level::warning)
          << "WARNING: Maximum k in CAMB input file is insufficient" << std::endl
          << "*        You therefore have zero power in some of your modes, which is almost certainly not what you want"
          << std::endl
          << "*        You need to generate a transfer function that reaches higher k." << std::endl
          << "*        The current grid reaches k = " << grid->getFourierKmax() << " h/Mpc" << std::endl;
      }
      return P;
    }

    //! Return the cosmology-dependent part of the normalisation of the power spectrum.
    void calculateOverallNormalization(const CosmologicalParameters<CoordinateType> &cosmology) {

      CoordinateType ourGrowthFactor = growthFactor(cosmology);
      CoordinateType growthFactorNormalized = ourGrowthFactor / growthFactor(cosmologyAtRedshift(cosmology, 0));

      CoordinateType sigma8PreNormalization = calculateLinearVarianceInSphere(8.);
      CoordinateType linearRenorm = (cosmology.sigma8 / sigma8PreNormalization);
      CoordinateType linearRenormFactor = linearRenorm * growthFactorNormalized;
      
      //used for computing RMS of Vbc field
      linearAmplitude = linearRenorm * linearRenorm;
      
      //used for computing power spectrum with correct backscaled amplitude
      amplitude = linearRenormFactor * linearRenormFactor;
    }

    //! Compute the variance in a spherical top hat window
    // Compute the linear-theory rms variance of the density field smoothed
    // with a spherical top–hat of radius R in real space.
    // This evaluates the standard integral:
    //
    //    σ^2(R) = (1 / 2π^2) ∫ dk  k^2  P(k)  |W(kR)|^2
    //
    // where W(kR) is the Fourier-space top–hat filter. The power spectrum
    // is obtained via the CAMB transfer function (t), assuming P(k) ∝ k^ns t^2.
    //
    // Returns σ(R), not σ^2.
    CoordinateType calculateLinearVarianceInSphere(
          CoordinateType radius,
          particle::species transferType = particle::species::all) const {
    
          CoordinateType s = 0., k, t;
    
          // Prefactor = 9 / (2π^2), coming from |W(kR)|^2 combined with
          // the 1/(2π^2) normalization of the variance integral.
          CoordinateType amp =
            static_cast<CoordinateType>(9) / static_cast<CoordinateType>(2) / M_PI / M_PI;
    
          // Upper integration bound.
          // Limit to both the table range and  ~200/R to avoid ringing and small-scale noise.
          // The 0.999999 avoids exactly hitting the table boundary.
          CoordinateType kmax =
            std::min(static_cast<CoordinateType>(kInterpolationPoints.back()),
                     static_cast<CoordinateType>(200.0) / radius) *
            static_cast<CoordinateType>(0.999999);
    
          // Lower integration bound.
          // Slightly above the minimum tabulated k to avoid interpolation edge effects.
          CoordinateType kmin =
            static_cast<CoordinateType>(kInterpolationPoints[0]) *
            static_cast<CoordinateType>(1.000001);
    
          // Uniform k–space sampling step for the discrete sum approximation.
          // 50,000 steps is chosen to ensure numerical stability.
          CoordinateType dk = (kmax - kmin) / static_cast<CoordinateType>(50000);
    
          // Retrieve the appropriate transfer function (baryon, CDM, total matter, etc.)
          auto &interpolator = this->speciesToTransferFunction.at(transferType);
    
          for (k = kmin; k < kmax; k += dk) {
            // Interpolate transfer function at k.
            // t encodes the linear evolution scaling applied to primordial modes.
            t = interpolator(k);
    
            const CoordinateType kr = k * radius;
    
            // Fourier-space spherical top-hat window:
            //      W(x) = 3 (sin x − x cos x) / x^3
            //
            // Here the factor of 3 has already been absorbed into `amp` above.
            const CoordinateType W =
              (std::sin(kr) - kr * std::cos(kr)) / (kr * kr * kr);

            s += std::pow(k, ns + static_cast<CoordinateType>(2)) * W * W * t * t;
          }
          // Multiply by dk and prefactor, then take square root to obtain σ(R)
          s = std::sqrt(s * amp * dk);
          return s;
    }

  };

} // namespace cosmology

#endif

#ifndef _CAMB_HPP_INCLUDED
#define _CAMB_HPP_INCLUDED

#include <cmath>
#include <memory>
#include <utility>
#include <map>

#include "src/cosmology/parameters.hpp"
#include "src/tools/numerics/interpolation.hpp"
#include "src/io/input.hpp"
#include "src/simulation/particles/particle.hpp"
#include "src/tools/logging.hpp"
#include "src/tools/lru_cache.hpp"

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

    using CacheKeyType = std::pair<std::weak_ptr<const grids::Grid<CoordinateType>>, particle::species>;

    //! A cache for previously calculated covariances. The key is a pair: (weak pointer to the grid, transfer fn)
    // mutable std::map<CacheKeyType, std::shared_ptr<FieldType>,
    //  CacheKeyComparator<CacheKeyType>> calculatedCovariancesCache;

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

      if(transferType == particle::species::whitenoise)
        return nullptr;

      if(lru_cache_size==0)
        return getPowerSpectrumForGridUncached(grid, transferType);

      auto cacheKey = std::make_pair(std::weak_ptr<const grids::Grid<CoordinateType>>(grid), transferType);

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

      P->forEachFourierCell([norm, this, transferType]
                              (std::complex<CoordinateType>, CoordinateType kx, CoordinateType ky,
                               CoordinateType kz) {
        CoordinateType k = sqrt(kx * kx + ky * ky + kz * kz);
        auto spec = std::complex<CoordinateType>((*this)(k, transferType) * norm, 0);
        return spec;
      });

      return P;

    }

  public:

    //! Return the box- and fft-dependent part of the normalisation of the power spectrum
    static CoordinateType getPowerSpectrumNormalizationForGrid(const grids::Grid<CoordinateType> &grid) {

      CoordinateType kw = 2. * M_PI / grid.thisGridSize;
      CoordinateType norm = kw * kw * kw / pow(2.f * M_PI, 3.f); //since kw=2pi/L, this is just 1/V_box

      // This factor Ncells was first needed when FFT normalisation changed from 1/N to 1/sqrt(N). This knowledge was previously
      // incorporated as a normalisation of the random draw rather than to the power spectrum. It makes more sense to have
      // it as a PS normalisation and restores coherence between the P(k) estimated from the field (e.g. delta dagger * delta)
      // and the theoretical P(k) calculated here. MR 2018
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
    PowerLawPowerSpectrum(const CosmologicalParameters<CoordinateType> &cosmology, CoordinateType amplitude)
    : ns(cosmology.ns), amplitude(amplitude) { }

    CoordinateType operator()(CoordinateType k, particle::species transferType) const override {
      if(k==0)
        return 0;
      else
        return amplitude * pow(k, ns);
    }

  };

  /*! \class CAMB
  * \brief Provides power spectra by using transfer functions from CAMB output
  */
  template<typename DataType>
  class CAMB : public PowerSpectrum<DataType> {
    using typename PowerSpectrum<DataType>::CoordinateType;

  protected:
    std::vector<double> kInterpolationPoints; //!< Wavenumbers read from CAMB file
    std::map<particle::species, std::vector<double>> speciesToInterpolationPoints; //!< Vector to store transfer functions

    const std::map<particle::species, size_t> speciesToCambColumn
      {{particle::species::dm, 1},
       {particle::species::baryon, 2},
       {particle::species::all, 6 // if using a single transfer function, use the column for total
     }};
      //!< Columns of CAMB that we request for DM and baryons respectively

    std::map<particle::species, tools::numerics::LogInterpolator<double>> speciesToTransferFunction; //!< Interpolation functions:
    CoordinateType amplitude; //!< Amplitude of the initial power spectrum
    CoordinateType ns;        //!< tensor to scalar ratio of the initial power spectrum
    mutable CoordinateType kcamb_max_in_file; //!< Maximum CAMB wavenumber. If too small compared to grid resolution, Meszaros solution will be computed

    CoordinateType isocurvatureTransferRescale; //!< Backscaling factor applied to transfer functions for alpha computation
    CoordinateType isocurvatureTargetRedshift;  //!< Target redshift used for isocurvature transfer-function rescaling

  public:
    //! Import data from CAMB file and initialise the interpolation functions used to compute the transfer functions:
    CAMB(const CosmologicalParameters<CoordinateType> &cosmology, const std::string &filename) {
      readLinesFromCambOutput(filename);
      for (auto i = speciesToInterpolationPoints.begin(); i != speciesToInterpolationPoints.end(); ++i) {
        this->speciesToTransferFunction[i->first].initialise(kInterpolationPoints, i->second);
      }
      ns = cosmology.ns;
      calculateOverallNormalization(cosmology);

      isocurvatureTargetRedshift = static_cast<CoordinateType>(cosmology::isocurvature_redshift);

      // Backscale transfer-function amplitudes from z=0 to the configured target redshift for alpha coefficient calculation.
      CoordinateType growth0 = growthFactor(cosmologyAtRedshift(cosmology, 0));
      CoordinateType growthiso = growthFactor(cosmologyAtRedshift(cosmology, isocurvatureTargetRedshift));
      isocurvatureTransferRescale = growthiso / growth0;

      const CoordinateType alpha = calculateAlphaCoefficientDiscrete();
      isocurvature_alpha() = static_cast<double>(alpha);
    }

    CoordinateType operator()(CoordinateType k, particle::species transferType) const override {
      CoordinateType linearTransfer;
      if (k != 0)
        linearTransfer = speciesToTransferFunction.at(transferType)(k);
      else
        linearTransfer = 0.0;

      if (k > kcamb_max_in_file) {
        kcamb_max_in_file = std::numeric_limits<CoordinateType>().max();
      }

      return amplitude * pow(k, ns) * linearTransfer * linearTransfer;
    }

    /*!
     * \brief Compute alpha = <δ_bc δ_m> / <δ_m^2> from the discrete CAMB transfer-function table.
     *
     * We interpret:
     *   T_bc(k) = T_b(k) - T_cdm(k)
     *   T_m (k) = T_all(k)  (CAMB "total matter" column configured via particle::species::all)
     *
     * Using the same primordial scaling as the rest of this class:
     *   P_ij(k) ∝ k^{ns} T_i(k) T_j(k)
     * and isotropic measure d^3k = 4π k^2 dk, the common constants cancel, giving
     *
     *   alpha = ∫ dk k^{ns+2} T_bc(k) T_m(k) / ∫ dk k^{ns+2} T_m(k)^2 .
     *
     * We evaluate the integrals with a trapezoidal rule in ln k:
     *   ∫ dk f(k) = ∫ dlnk [k f(k)] .
     */
    CoordinateType calculateAlphaCoefficientDiscrete() const {
      logging::entry()
        << "Calculating alpha coefficient with baryon, CDM, and matter transfers "
        << "backscaled to z=" << isocurvatureTargetRedshift << std::endl;
    
      auto it_c = speciesToInterpolationPoints.find(particle::species::dm);
      auto it_b = speciesToInterpolationPoints.find(particle::species::baryon);
      auto it_m = speciesToInterpolationPoints.find(particle::species::all);
    
      if (it_c == speciesToInterpolationPoints.end() ||
          it_b == speciesToInterpolationPoints.end() ||
          it_m == speciesToInterpolationPoints.end()) {
        throw std::runtime_error(
          "Cannot compute alpha: required CAMB transfer columns (dm, baryon, all) are missing."
        );
      }
    
      const auto &kcamb = kInterpolationPoints;
      const auto &Tc = it_c->second;
      const auto &Tb = it_b->second;
      const auto &Tm = it_m->second;
    
      size_t n = kcamb.size();
      if (Tc.size() < n) n = Tc.size();
      if (Tb.size() < n) n = Tb.size();
      if (Tm.size() < n) n = Tm.size();
    
      if (n < 2) {
        throw std::runtime_error(
          "Cannot compute alpha: CAMB transfer table has insufficient points."
        );
      }
    
      const CoordinateType g = isocurvatureTransferRescale;
    
      CoordinateType num = 0;
      CoordinateType den = 0;
    
      for (size_t i = 0; i + 1 < n; ++i) {
        const CoordinateType k1 = static_cast<CoordinateType>(kcamb[i]);
        const CoordinateType k2 = static_cast<CoordinateType>(kcamb[i + 1]);
    
        if (k1 <= 0 || k2 <= 0) continue;
    
        const CoordinateType dlnk = std::log(k2 / k1);
    
        const CoordinateType Tb1 = g * static_cast<CoordinateType>(Tb[i]);
        const CoordinateType Tb2 = g * static_cast<CoordinateType>(Tb[i + 1]);
    
        const CoordinateType Tc1 = g * static_cast<CoordinateType>(Tc[i]);
        const CoordinateType Tc2 = g * static_cast<CoordinateType>(Tc[i + 1]);
    
        const CoordinateType Tm1 = g * static_cast<CoordinateType>(Tm[i]);
        const CoordinateType Tm2 = g * static_cast<CoordinateType>(Tm[i + 1]);
    
        const CoordinateType Tbc1 = Tb1 - Tc1;
        const CoordinateType Tbc2 = Tb2 - Tc2;
    
        const CoordinateType w1 = std::pow(k1, ns + 3);
        const CoordinateType w2 = std::pow(k2, ns + 3);
    
        const CoordinateType fnum1 = w1 * Tbc1 * Tm1;
        const CoordinateType fnum2 = w2 * Tbc2 * Tm2;
    
        const CoordinateType fden1 = w1 * Tm1 * Tm1;
        const CoordinateType fden2 = w2 * Tm2 * Tm2;
    
        num += static_cast<CoordinateType>(0.5) * (fnum1 + fnum2) * dlnk;
        den += static_cast<CoordinateType>(0.5) * (fden1 + fden2) * dlnk;
      }
    
      if (den == static_cast<CoordinateType>(0)) {
        throw std::runtime_error(
          "Cannot compute alpha: denominator integral is zero."
        );
      }
    
      const CoordinateType alpha = num / den;
    
      logging::entry()
        << "Computed alpha coefficient = " << alpha << std::endl;
    
      return alpha;
    }



  protected:

      //! \brief This function imports data from a CAMB file, supplied as a file-name string argument (filename).
      //! Both pre-2015 and post-2015 formats can be used, and the function will detect which.
      void readLinesFromCambOutput(std::string filename) {
        kInterpolationPoints.clear();
        speciesToInterpolationPoints.clear();

        // Dealing with the update of the CAMB TF output. Both are kept for backward compatibility.
        const int c_old_camb = 7; // number of columns in camb transfer function pre 2015
        const int c_new_camb = 13; // number of columns in camb transfer function post 2015
        int numCols;
        size_t j;

        // Import data from CAMB file:
        std::vector<double> input;
        io::getBufferIgnoringColumnHeaders(input, filename);

        // Check whether the input file is in the pre-2015 or post-2015 format (and throw an error if it is neither).
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


        CoordinateType transferNormalisation = input[1]; // to normalise CAMB transfer function so T(0)= 1, doesn't matter if we normalise here in terms of accuracy, but feels more natural
        // Copy file into vectors. Normalise both so that the DM tranfer function starts at 1.
        for (j = 0; j < input.size() / numCols; j++) {
          if (input[numCols * j] > 0) {
            // hard-coded to first two columns of CAMB file -
            kInterpolationPoints.push_back(CoordinateType(input[numCols * j]));
            for (auto i = speciesToCambColumn.begin(); i != speciesToCambColumn.end(); ++i) {
              speciesToInterpolationPoints[i->first].push_back(CoordinateType(input[numCols * j + i->second]) / transferNormalisation);
            }
          } else continue;
        }

        kcamb_max_in_file = kInterpolationPoints.back();

      }

    //! Calculate the theoretical power spectrum for a given grid
    std::shared_ptr<fields::Field<DataType, CoordinateType>>
    getPowerSpectrumForGridUncached(std::shared_ptr<const grids::Grid<CoordinateType>> grid,
                                    particle::species transferType = particle::species::dm) const override {

      auto P = PowerSpectrum<DataType>::getPowerSpectrumForGridUncached(grid, transferType);

      if (kcamb_max_in_file == std::numeric_limits<CoordinateType>().max()) {
        logging::entry(logging::level::warning) << "WARNING: Maximum k in CAMB input file is insufficient" << std::endl
                  << "*        You therefore have zero power in some of your modes, which is almost certainly not what you want" << std::endl
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
        CoordinateType linearRenormFactor = (cosmology.sigma8 / sigma8PreNormalization) * growthFactorNormalized;

        amplitude = linearRenormFactor * linearRenormFactor;
      }
    
      //! Compute the variance in a spherical top hat window
      CoordinateType calculateLinearVarianceInSphere(CoordinateType radius,
                                                     particle::species transferType = particle::species::all) const {

        CoordinateType s = 0., k, t;

        CoordinateType amp = 9. / 2. / M_PI / M_PI;
        CoordinateType kmax = std::min(kInterpolationPoints.back(), 200.0 / radius)*0.999999;
        CoordinateType kmin = kInterpolationPoints[0]*1.000001;

        CoordinateType dk = (kmax - kmin) / 50000.;
        auto &interpolator = this->speciesToTransferFunction.at(transferType);
        for (k = kmin; k < kmax; k += dk) {


          t = interpolator(k);

          // Multiply power spectrum by the fourier transform of the spherical top hat, to give the fourier transform
          // of the averaged (convolved) power spectrum over the sphere.
          s += pow(k, ns + 2.) *
               ((sin(k * radius) - k * radius * cos(k * radius)) / ((k * radius) * (k * radius) * (k * radius))) *
               ((sin(k * radius) - k * radius * cos(k * radius)) / ((k * radius) * (k * radius) * (k * radius))) * t * t;

        }


        s = sqrt(s * amp * dk);
        return s;

      }
    
  };

}

#endif
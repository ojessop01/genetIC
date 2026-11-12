#ifndef IC_RANDOMFIELDGENERATOR_HPP
#define IC_RANDOMFIELDGENERATOR_HPP

#include <gsl/gsl_rng.h> //link -lgsl and -lgslcblas at the very end
#include <gsl/gsl_randist.h> //for the gaussian (and other) distributions
#include <gsl/gsl_errno.h>
#include <gsl/gsl_spline.h>
#include <omp.h>

#include "src/simulation/grid/grid.hpp"

namespace fields {

  /*! \class RandomFieldGenerator
      \brief This class handles drawing random white noise across all grids in a multi-level field.

      Essentially, the class stores GSL random number generators, together with a list of options for how
      the random numbers should be drawn. It also keeps track of the seed used, and the field for which it
      is generating random data.

      RandomFieldGenerator allows numbers to be drawn both in parallel and in series, depending on the options chosen.
      However, by construction, it does not allow reseeding of an already seeded field.
    */
  template<typename DataType>
  class RandomFieldGenerator {
  protected:
    using FloatType = tools::datatypes::strip_complex<DataType>;

    gsl_rng *randomState; //!< Pointer to the random number generator
    const gsl_rng_type *randomNumberGeneratorType; //!< pointer to the type of random number generator being used.
    bool drawInFourierSpace; //!< If true, random numbers are drawn in Fourier space.
    bool reverseRandomDrawOrder; //!< If true, order in which random numbers for complex numbers is drawn is reversed.
    bool seeded; //!< True if the random number generator has already been seeded
    bool parallel; //!< True if we want to draw random numbers in parallel
    bool drawShellEdges = true; //!< If false, reproduce the pre-fix Fourier draw, which never drew the shell-edge modes
    unsigned long baseSeed; //!< Stores the last seed used.
    MultiLevelField <DataType> &field; //!< Reference to the multilevel field we are drawing random numbers for.


  public:
    //! Constructor with known multi-level field.
    RandomFieldGenerator(MultiLevelField <DataType> &field_, unsigned long seed = 0) :
      field(field_) {
      randomNumberGeneratorType = gsl_rng_ranlxs2; // shouldn't this be gsl_rng_ranlxd2 for FloatType = double? -> it's single precision for compatibility with previous versions!
      randomState = gsl_rng_alloc(randomNumberGeneratorType); // this allocates memory for the generator with type T
      gsl_rng_set(randomState, seed);
      this->baseSeed = seed;
      drawInFourierSpace = false;
      seeded = false;
      parallel = false;
    }

    //! Copy constructor
    RandomFieldGenerator(const RandomFieldGenerator &copy) : field(copy.field) {
      // Copy across old variables:
      seeded = copy.seeded;
      baseSeed = copy.baseSeed;
      parallel = copy.parallel;
      drawShellEdges = copy.drawShellEdges;
      randomNumberGeneratorType = copy.randomNumberGeneratorType;
      drawInFourierSpace = copy.drawInFourierSpace;
      reverseRandomDrawOrder = copy.reverseRandomDrawOrder;

      // Construct our copy's own generator (resizing a vector of randomFieldGenerators will
      // delete the object randomState points to otherwise, so each copy needs its own instance!):
      randomState = gsl_rng_alloc(randomNumberGeneratorType);
      gsl_rng_set(randomState, copy.baseSeed);

    }

    //! Constructor, using settings from existing generator but tying to new field (used for splicing)
    RandomFieldGenerator(MultiLevelField <DataType> &field_, const RandomFieldGenerator &copy) : field(field_) {
      seeded = false;
      baseSeed = copy.baseSeed;
      parallel = copy.parallel;
      drawShellEdges = copy.drawShellEdges;
      randomNumberGeneratorType = copy.randomNumberGeneratorType;
      drawInFourierSpace = copy.drawInFourierSpace;
      reverseRandomDrawOrder = copy.reverseRandomDrawOrder;
      randomState = gsl_rng_alloc(randomNumberGeneratorType);
      gsl_rng_set(randomState, copy.baseSeed);
    }

    //! Destructor - frees the random state
    virtual ~RandomFieldGenerator() {
      gsl_rng_free(randomState);
    }

    using RefFieldType = std::vector<DataType> &;
    using FieldType = std::remove_reference_t<RefFieldType>;


    //! Sets drawInFourierSpace to true or false
    void setDrawInFourierSpace(bool value) {
      drawInFourierSpace = value;
    }

    //! Sets parallel to true or false
    void setParallel(bool value) {
      parallel = value;
    }

    //! Sets reverseRandomDrawOrder to true or false
    void setReverseRandomDrawOrder(bool value) {
      reverseRandomDrawOrder = value;
    }

    /*! \brief If false, leave the shell-edge Fourier modes at zero, as genetIC did before they were drawn.

        Only useful to reproduce initial conditions made with earlier versions bit for bit; see
        drawRandomForSpecifiedGridFourier.
    */
    void setDrawShellEdges(bool value) {
      drawShellEdges = value;
    }

    //!\brief Seeds the random number generator with the specified seed.
    /*!
        \param seed - seed we wish to use

        Note that we cannot seed more than once, or an error will be generated.
    */
    void seed(unsigned long seed) {
      if (seeded) {
        throw std::runtime_error("The random number generator has already been seeded");
      } else {
        gsl_rng_set(randomState, seed);
        this->baseSeed = seed;
        seeded = true;
      }
    }

    //! Draws random numbers for multi level field.
    void draw() {
      if (!seeded)
        throw std::runtime_error("The random number generator has not been seeded");
      for (size_t i = 0; i < field.getNumLevels(); ++i) {
        auto &fieldOnGrid = field.getFieldForLevel(i);
        if(i==0)
          logging::entry() << "Drawing random numbers (base grid)" << std::endl;
        else
          logging::entry() << "Drawing random numbers (zoom level " << i << ")" << std::endl;

        if (drawInFourierSpace) {
          fieldOnGrid.toFourier();
          drawRandomForSpecifiedGridFourier(fieldOnGrid);
        } else {
          drawRandomForSpecifiedGrid(fieldOnGrid);
        }
      }
    }

  protected:

    //! Draws a random number for a given Fourier mode.
    void drawOneFourierMode(Field <DataType> &field, int k1, int k2, int k3,
                            FloatType norm, gsl_rng *localRandomState) {

      // these need to be initialized in explicit order - can't leave it to compilers
      // to choose as they choose differently...
      FloatType a = norm * gsl_ran_gaussian_ziggurat(localRandomState, 1.);
      FloatType b = norm * gsl_ran_gaussian_ziggurat(localRandomState, 1.);

      if (reverseRandomDrawOrder)
        field.setFourierCoefficient(k1, k2, k3, tools::datatypes::ensure_complex<DataType>(b, a));
      else
        field.setFourierCoefficient(k1, k2, k3, tools::datatypes::ensure_complex<DataType>(a, b));

      int nyquist = int(field.getGrid().size) / 2;

      if (k1 == 0 || k1 == nyquist || k2 == 0 || k2 == nyquist || k3 == 0 || k3 == nyquist) {
        // Due to poor original implementation of drawRandomForSpecifiedGridFourier (which we're now stuck with
        // for historical compatibility), we need to ensure the _last_ mode written to a field (which may, due to
        // symmetries in the Fourier coeffs at 0 or nyquist modes, overwrite a previous draw) persists.
        //
        // The above condition probably catches too many cases, but if there is a risk, let's also explicitly write
        // the related coeff.
        if (reverseRandomDrawOrder)
          field.setFourierCoefficient(-k1, -k2, -k3, tools::datatypes::ensure_complex<DataType>(b, -a));
        else
          field.setFourierCoefficient(-k1, -k2, -k3, tools::datatypes::ensure_complex<DataType>(a, -b));
      }
    }


    /*! \brief Draw random white noise in real space.
       *
       *  Kept for historical compatibility, even though the recommended approach is
       * to seed in Fourier space (routine drawRandomForSpecifiedGridFourier, accessed using command
       * seedfourier in the paramfile)
       *
       */
    void drawRandomForSpecifiedGrid(Field <DataType> &field) {


      field.setFourier(false);
      auto &g = field.getGrid();
      auto &fieldData = field.getDataVector();
      size_t nPartTotal = g.size3;

      // N.B. DO NOT PARALLELIZE this loop - want things to be done in a reliable order
      for (size_t i = 0; i < nPartTotal; i++) {
        fieldData[i] = gsl_ran_gaussian_ziggurat(randomState, 1.);
      }

      field.toFourier();
      tools::set_zero(fieldData[0]);

    }

    /*! \brief Draw the modes on the edges of cubic shell ks that the face loops cannot reach.

        The face loops in drawRandomForSpecifiedGridFourier draw every mode with a component equal to +ks whose other
        two components lie in [-ks, ks). A real field's mode k is the conjugate of -k, so a mode is covered if k or -k
        is drawn. That misses exactly the modes where +ks (or -ks) occurs in two or three components:
        (ks, ks, x), (ks, x, ks) and (x, ks, ks) with -ks < x < ks, and the corner (ks, ks, ks). Those are 6 ks - 2
        modes per shell (about 2% of all modes on a 64^3 grid, and more than 30% of the largest-scale modes), and
        before this was fixed they were left at zero.
    */
    void drawShellEdgeModes(Field <DataType> &field, int ks, FloatType norm, gsl_rng *localRandomState) {
      if (ks == 0)
        return;
      for (int x = -ks + 1; x < ks; x++) {
        drawOneFourierMode(field, ks, ks, x, norm, localRandomState);
        drawOneFourierMode(field, ks, x, ks, norm, localRandomState);
        drawOneFourierMode(field, x, ks, ks, norm, localRandomState);
      }
      drawOneFourierMode(field, ks, ks, ks, norm, localRandomState);
    }

    /*! \brief Draw random white noise in Fourier space.
        \param field - field to draw for
        Draws will potentially be in parallel if this option has been specified.

        Modes are drawn shell by shell (shell ks holds the modes whose largest |component| is ks), so for a given box
        and seed the same modes are drawn at any grid resolution. The shell-edge modes are drawn after the other draws
        of their shell (parallel) or from a separate per-shell stream (serial), so that adding them did not change the
        value of any mode that earlier versions drew. Modes with a component at the Nyquist frequency are not drawn.
    */
    void drawRandomForSpecifiedGridFourier(Field <DataType> &field) {

      std::vector<DataType> &vec = field.getDataVector();

      std::fill(vec.begin(), vec.end(), DataType(0));

      const grids::Grid<FloatType> &g = field.getGrid();

      tools::progress::ProgressBar pb("");

      FloatType sigma = 1.0 / sqrt(2.0);


      // Both approaches to making the random field below do it in square k-shells, in order of increasing |k|, so that
      // resolution can be scaled and we still get the 'same' field.
      //
      // The original, serial approach uses a single seed. The new parallel approach uses a seed for each shell in
      // k space. Note that one therefore gets different fields when running in parallel vs in serial.

      if (parallel) {
        // N.B. Parallel implementation produces results that are incompatible with the original serial
        // implementation

#pragma omp parallel for schedule(dynamic)
        for (int ks = 0; ks < int(g.size / 2); ks++) {
          gsl_rng *localRandomState = gsl_rng_alloc(randomNumberGeneratorType);
          gsl_rng_set(localRandomState, baseSeed + ks);
#ifdef OPENMP
          if (omp_get_thread_num() == 0)
#endif
            pb.setProgress(float(ks * ks) * (ks * 8) / g.size3);
          for (int k1 = -ks; k1 < ks; k1++) {
            for (int k2 = -ks; k2 < ks; k2++) {
              drawOneFourierMode(field, ks, k1, k2, sigma, localRandomState);
              drawOneFourierMode(field, k1, ks, k2, sigma, localRandomState);
              drawOneFourierMode(field, k1, k2, ks, sigma, localRandomState);
            }
          }
          if (drawShellEdges)
            drawShellEdgeModes(field, ks, sigma, localRandomState);
          gsl_rng_free(localRandomState);

        }

      } else {
        // N.B. This is the original way of doing things that does not allow for parallelization
        for (int ks = 0; ks < int(g.size / 2); ks++) {
          pb.setProgress(float(ks * ks) * (ks * 8) / g.size3);
          for (int k1 = -ks; k1 < ks; k1++) {
            for (int k2 = -ks; k2 < ks; k2++) {
              drawOneFourierMode(field, ks, k1, k2, sigma, randomState);
              drawOneFourierMode(field, k1, ks, k2, sigma, randomState);
              drawOneFourierMode(field, k1, k2, ks, sigma, randomState);
            }
          }
        }
        // All shells share one random stream here, so the edges cannot be drawn from it without changing the face
        // modes that earlier versions drew. Each shell's edges get their own stream instead, which also keeps them
        // independent of the grid resolution.
        if (drawShellEdges) {
          gsl_rng *edgeRandomState = gsl_rng_alloc(randomNumberGeneratorType);
          for (int ks = 1; ks < int(g.size / 2); ks++) {
            gsl_rng_set(edgeRandomState, baseSeed + ks);
            drawShellEdgeModes(field, ks, sigma, edgeRandomState);
          }
          gsl_rng_free(edgeRandomState);
        }
      }


    }


  };

}


#endif //IC_RANDOMFIELDGENERATOR_HPP

#include <iostream>

#include <gnuplot-iostream.h>

#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

#include <vector>

#include <runningstats/runningstats.h>
namespace rs = runningstats;

#include <ceres/ceres.h>

#include <libraw/libraw.h>

rs::QuantileStats<float> readout_times;
rs::QuantileStats<float> readout_speeds;

double start = 0.05;
double end = 0.95;

cv::Mat read_raw2(const std::string &filename) {
    LibRaw RawProcessor;

    bool verbose = true;
    auto& S = RawProcessor.imgdata.sizes;
    auto& OUT = RawProcessor.imgdata.params;

    int ret;
    if ((ret = RawProcessor.open_file(filename.c_str())) != LIBRAW_SUCCESS)
    {
        throw std::runtime_error(std::string("Cannot open file ") + filename + ", "
                                 + libraw_strerror(ret) + "\r\n");
    }
    if (verbose) {
        printf("Image size: %dx%d\nRaw size: %dx%d\n", S.width, S.height, S.raw_width, S.raw_height);
        printf("Margins: top=%d, left=%d\n", S.top_margin, S.left_margin);
    }

    if ((ret = RawProcessor.unpack()) != LIBRAW_SUCCESS) {
        throw std::runtime_error(std::string("Cannot unpack file ") + filename + ", "
                                 + libraw_strerror(ret) + "\r\n");
    }

    if (verbose)
        printf("Unpacked....\n");

    if (!(RawProcessor.imgdata.idata.filters || RawProcessor.imgdata.idata.colors == 1)) {
        throw std::runtime_error(
                    std::string("Only Bayer-pattern RAW files supported, file ") + filename
                    + " seems to have a different pattern.\n");
    }

    cv::Mat result(S.raw_height, S.raw_width, CV_16UC1);

    for (int jj = 0, global_counter = 0; jj < S.raw_height ; jj++) {
        unsigned short * row = result.ptr<unsigned short>(jj);
        for (int ii = 0; ii < S.raw_width; ++ii, ++global_counter) {
            row[ii] = RawProcessor.imgdata.rawdata.raw_image[global_counter];
        }
    }


    return result;
}

cv::Mat_<uint16_t> read_raw(std::string const& fn) {
    LibRaw rawProcessor;
    libraw_processed_image_t *tmpImg;

    if(rawProcessor.open_file(fn.c_str()) != LIBRAW_SUCCESS) {
        throw std::runtime_error("file not readable");
    }
    rawProcessor.unpack(); // decode bayer data
    rawProcessor.dcraw_process(); // white balance, color interpolation, color space conversion
    rawProcessor.dcraw_make_mem_image(); // gamma correction, image rotation, 3-component RGB bitmap creation

    cv::Mat_<cv::Vec3w> img = cv::Mat(rawProcessor.imgdata.sizes.height, rawProcessor.imgdata.sizes.width, CV_16UC3, rawProcessor.imgdata.image);
    cv::imshow("img", img);
    while (true) {
        char key = cv::waitKey(0);
        if ('q' == key) {
            break;
        }
    }

    cv::Mat_<uint16_t> result(img.size(), uint16_t(0));
    cv::cvtColor(img, result, cv::COLOR_RGB2GRAY);
    rawProcessor.recycle();
    return result;
}

cv::Mat_<uint16_t> read_any_image(std::string const& fn) {
    try {
        cv::Mat_<uint16_t> img = read_raw2(fn);
        return img;
    }
    catch(std::exception const& e) {
        std::cout << "Libraw failed reading the image: " << std::endl << e.what() << std::endl;
    }
    return cv::imread(fn, cv::IMREAD_GRAYSCALE | cv::IMREAD_ANYDEPTH);
}

/**
 * @brief sum_rows_normalize takes an image and produces a vector with one entry per row of the input image.
 * The entry is the average of the corresponding row, normalized by the maximum of all entries.
 * @param img
 * @return
 */
std::vector<float> sum_rows_normalize(cv::Mat1f const& img) {
    std::vector<float> result(img.rows, float(0));
    float max = - std::numeric_limits<float>::max();
    for (int ii = 0; ii < img.rows; ++ii) {
        for (int jj = 0; jj < img.cols; ++jj) {
            result.at(ii) += img(ii,jj);
        }
        max = std::max(max, result.at(ii));
    }
    for (float& val : result) {
        val /= max;
    }
    return result;
}

std::vector<float> moving_normalization(std::vector<float> const& input, size_t const num_pts, size_t const step) {
    std::vector<float> result(input);
    for (size_t ii = 0; ii < input.size(); ii += step) {
        float max = input[ii];
        for (size_t jj = ii; jj < input.size() && jj < ii+num_pts; ++jj) {
            max = std::max(max, input.at(jj));
        }
        for (size_t jj = ii; jj < input.size() && jj < ii+num_pts; ++jj) {
            result.at(jj) = input.at(jj)/max;
        }
    }
    return result;
}


struct SinCost {
    double t;
    double val;

    template<class T>
    static T mysin(T val) {
        return (ceres::sin(val) + T(1))/T(2);
    }

    template<class T>
    bool operator()(
            T const * const wavelength,
            T const * const offset,
            T const * const low,
            T const * const amplitude,
            T * residuals) const {
        T const factor(2*M_PI);
        residuals[0] = -T(val) + low[0] + amplitude[0] * mysin(T(t)*factor/wavelength[0] - offset[0]);
        return true;
    }

    static ceres::CostFunction* create(double const _t, double const _val) {
        return new ceres::AutoDiffCostFunction<SinCost, 1, 1, 1, 1, 1>(
                    new SinCost{_t, _val}
                    );
    }
};

struct SinResult {
    double wavelength = 1;
    double offset = 0;
    double low = 0;
    double amplitude = 1;

    /**
     * @brief Number of rows of the image. This is used for calculating the readout speed.
     */
    int num_rows = -1;

    /**
     * @brief Frequency of the test-LED used when capturing the test images. This is used for calculating readout speed.
     */
    double led_freq = 4'000;

    /**
     * @brief Lower threshold when counting rising edges.
     */
    float th_low;

    /**
     * @brief Higher threshold when counting rising edges.
     */
    float th_high;

    std::vector<int> rising_flanks;

    std::string gnuplot_function() const {
        std::stringstream out;
        out << "f(x) = " << low + amplitude/2 << "+" << amplitude/2 << "*sin(x*" << 2*M_PI/wavelength << "-" << offset << ")";
        return out.str();
    }

    double get_readout_time_ms() const {
        return double(num_rows)/wavelength/led_freq * 1'000;
    }

    double get_readout_speed() const {
        return wavelength * led_freq / 1'000;
    }

    std::string print() const {
        std::stringstream out;
        out << "wavelength:    " << wavelength << std::endl
            << "readout speed: " << get_readout_speed() << " rows per ms (assuming a LED frequency of " << led_freq << "Hz)" << std::endl
            << "readout time:  " << get_readout_time_ms() << "ms" << std::endl
            << "offset:        " << offset << std::endl
            << "low:           " << low << std::endl
            << "amplitude:     " << amplitude << std::endl
            << "function:      " << gnuplot_function() << std::endl;
        return out.str();
    }
};

SinResult compute_freq(std::vector<float> const& values) {
    rs::QuantileStats<float> stats;
    stats.push(values);

    for (double q : {.01, .02, .05, .25, .5, .75, .95, .98, .99}) {
        std::cout << "q " << q << ": " << stats.getQuantile(q) << std::endl;
    }

    SinResult res;
    res.th_high = stats.getQuantile(.6);
    res.th_low = stats.getQuantile(.4);

    int first_rising = -1;
    int last_rising = -1;
    int last_high = -1;
    int last_low = -1;
    bool was_low = false;

    int count_rising = -1;
    for (int ii = start*double(values.size()); ii < end*double(values.size()); ++ii) {
        if (values.at(ii) < res.th_low) {
            last_low = ii;
            was_low = true;
        }
        if (values.at(ii) > res.th_high) {
            last_high = ii;
            if (last_low > 0 && first_rising < 0) {
                res.rising_flanks.push_back(ii);
                first_rising = ii;
            }
            if (was_low) {
                res.rising_flanks.push_back(ii);
                last_rising = ii;
                count_rising++;
                was_low = false;
            }
        }
    }

    res.num_rows = values.size();
    res.wavelength = double(last_rising - first_rising) / count_rising;
    res.offset = first_rising;

    ceres::Problem problem;
    for (int tt = start*double(values.size()); tt < end*double(values.size()); ++tt) {
        problem.AddResidualBlock(
                    SinCost::create(tt, values[tt]),
                    nullptr, // L2 cost
                    &res.wavelength,
                    &res.offset,
                    &res.low,
                    &res.amplitude);
    }
    ceres::Solver::Options ceres_opts;
    ceres::Solver::Summary summary;

    ceres_opts.minimizer_progress_to_stdout = false;
    ceres_opts.linear_solver_type = ceres::DENSE_QR;
    ceres_opts.max_num_iterations = 3'000;
    double const tol_factor = 1e-3;
    ceres_opts.function_tolerance *= tol_factor;
    ceres_opts.gradient_tolerance *= tol_factor;
    ceres_opts.parameter_tolerance *= tol_factor;

    std::cout << "Initial guess: " << std::endl << res.print() << std::endl;

    problem.SetParameterBlockConstant(&res.wavelength);
    problem.SetParameterBlockConstant(&res.low);
    problem.SetParameterBlockConstant(&res.amplitude);
    std::cout << "Solving for offset: " << std::endl;
    ceres::Solve(ceres_opts, &problem, &summary);
    std::cout << "Intermediate result: " << std::endl << res.print() << std::endl;

    problem.SetParameterBlockVariable(&res.low);
    problem.SetParameterBlockVariable(&res.amplitude);
    std::cout << "Solving for low and amplitude: " << std::endl;
    ceres::Solve(ceres_opts, &problem, &summary);
    std::cout << "Intermediate result: " << std::endl << res.print() << std::endl;

    problem.SetParameterBlockVariable(&res.wavelength);
    std::cout << "Solving for wavelength:" << std::endl;
    ceres::Solve(ceres_opts, &problem, &summary);
    std::cout << "Final result: " << std::endl << res.print() << std::endl;

    readout_speeds.push(res.get_readout_speed());
    readout_times.push(res.get_readout_time_ms());

    return res;
}

struct EnvelopeCost {
    double t;
    double val;
    size_t n_params;

    /**
     * @brief The maximum possible number of parameters must be known at compile time.
     */
    static constexpr size_t max_params = 10;

    template<class T>
    /**
     * @brief horner evalutes a polynome of the form p[0] + xp[1] + x²p[2]... using Horner's scheme
     * @param x value x
     * @param params parameters of the polynomial
     * @param n_params number of parameters of the polynomial
     * @return result of the polynomial evaluation
     */
    static T horner(T const& x, T const * const params, size_t const n_params) {
        T result(0);
        for (size_t ii = 1; ii <= n_params; ++ii) {
            result = x*result + params[n_params-ii];
        }
        return result;
    }

    template<class T>
    bool operator() (T const * const params, T * residuals) const {
        residuals[0] = horner(T(t), params, n_params);
        return T(val) < residuals[0];
    }

    static ceres::CostFunction* create(double const _t, double const _val, size_t const _n_params) {
        return new ceres::AutoDiffCostFunction<EnvelopeCost, 1, max_params>(
                    new EnvelopeCost{_t, _val, _n_params}
                    );
    }

    static double eval(std::vector<double> const& params, double const t) {
        double result = 0;
        EnvelopeCost{t, 0, params.size()}(params.data(), &result);
        return result;
    }
};

std::vector<float> normalize_upper_envelope(std::vector<float> const& values, size_t const num_params) {
    ceres::Problem problem;

    std::vector<double> params(EnvelopeCost::max_params, double(0));
    params[0] = 1.1;

    for (size_t ii = 0; ii < values.size(); ++ii) {
        problem.AddResidualBlock(
                    EnvelopeCost::create(ii, values[ii], num_params),
                    nullptr,
                    params.data());
    }

    ceres::Solver::Options ceres_opts;
    ceres::Solver::Summary summary;

    ceres_opts.minimizer_progress_to_stdout = true;
    ceres_opts.linear_solver_type = ceres::DENSE_QR;
    ceres_opts.max_num_iterations = 3'000;
    double const tol_factor = 1e-3;
    ceres_opts.function_tolerance *= tol_factor;
    ceres_opts.gradient_tolerance *= tol_factor;
    ceres_opts.parameter_tolerance *= tol_factor;

    ceres::Solve(ceres_opts, &problem, &summary);
    std::cout << summary.BriefReport() << std::endl;

    for (size_t ii = 0; ii < num_params; ++ii) {
        std::cout << params[ii] << "*x**" << ii;
    }
    std::cout << std::endl;

    return values;
}

void process_file(std::string const fn) {
    std::cout << "Processing file " << fn << std::endl;
    cv::Mat_<uint16_t> img = read_any_image(fn);
    cv::GaussianBlur(img, img, cv::Size(), 3, 3);
    std::cout << "Size: " << img.size() << std::endl;
    for (size_t ii = 0; ii < 15; ++ii) {
        cv::createCLAHE(4.0, cv::Size(4,4))->apply(img, img);
    }
    /*
    cv::imshow("img", img);
    while (true) {
        char key = cv::waitKey(0);
        if ('q' == key) {
            break;
        }
    }
    */

    std::vector<float> sums = sum_rows_normalize(img);
    SinResult res = compute_freq(sums);

    std::string const plot_prefix = fn +  "-plot";

    gnuplotio::Gnuplot gpl("tee " + plot_prefix + ".gpl | gnuplot -persist");
    gpl << "set term png; set output '" << plot_prefix << ".png';\n";
    gpl << "set xlabel 'row'; set ylabel 'brightness';\n";
    gpl << "set samples 10000;\n";
    gpl << "set key horiz out;\n";
    for (int pos : res.rising_flanks) {
        gpl << "set arrow from " << pos << ", graph 0 to " << pos << ", graph 1 nohead;\n";
    }
    gpl << res.gnuplot_function() << ";\n";
    gpl << "plot " << gpl.file(sums, plot_prefix + "-raw.data") << " w l title 'raw', f(x) w l title 'fit', "
        << res.th_low << " w l title 'low', " << res.th_high << " w l title 'high'";

    std::cout << std::endl << std::endl;

}

int main(int argc, char ** argv)  {

    for (int ii = 1; ii < argc; ++ii) {
        process_file(argv[ii]);
    }

    std::cout << "Readout time stats: " << readout_times.print() << std::endl;
    std::cout << "Readout speed stats: " << readout_speeds.print() << std::endl;

}

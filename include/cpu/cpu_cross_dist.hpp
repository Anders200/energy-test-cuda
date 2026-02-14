#ifndef ENERGY_TEST_CROSS_HPP
#define ENERGY_TEST_CROSS_HPP

#include <cmath>
#include <vector>
#include <algorithm>
#include <random>
#include <numeric>

namespace CPU_cross_dist {

    struct PermutationStats {
        double observed_statistic = 0.0;
        std::vector<double> permuted_statistics; // length=permutations
    };

    template <typename Point>
    double euclidean_dist(const Point& a, const Point& b) {
        double sum = 0.0;
        size_t dims = a.size();
        for (size_t i = 0; i < dims; ++i) {
            double diff = a[i] - b[i];
            sum += diff * diff;
        }
        return std::sqrt(sum);
    }

    // ============================================================
    // Pooled distance matrix helpers
    // ============================================================

    // Build a single pooled distance matrix D for pooled = [X..., Y...].
    // nX is the number of points belonging to X, N = pooled.size().
    template <typename Container>
    std::vector<std::vector<double>> compute_pooled_distance_matrix(
        const Container& X,
        const Container& Y
    ) {
        const size_t nX = X.size();
        const size_t nY = Y.size();
        const size_t N = nX + nY;

        Container pooled = X;
        pooled.insert(pooled.end(), Y.begin(), Y.end());

        std::vector<std::vector<double>> D(N, std::vector<double>(N, 0.0));
        for (size_t i = 0; i < N; ++i) {
            for (size_t j = i + 1; j < N; ++j) {
                const double d = euclidean_dist(pooled[i], pooled[j]);
                D[i][j] = D[j][i] = d;
            }
        }
        return D;
    }

    // Energy statistic given pooled distance matrix D and X/Y split at nX.
    inline double energy_statistic_from_pooled(
        const std::vector<std::vector<double>>& D,
        size_t nX
    ) {
        const size_t N = D.size();
        const size_t nY = N - nX;

        double sumXX = 0.0;
        for (size_t i = 0; i < nX; ++i)
            for (size_t j = i + 1; j < nX; ++j)
                sumXX += D[i][j];

        double sumYY = 0.0;
        for (size_t i = nX; i < N; ++i)
            for (size_t j = i + 1; j < N; ++j)
                sumYY += D[i][j];

        double sumXY = 0.0;
        for (size_t i = 0; i < nX; ++i)
            for (size_t j = nX; j < N; ++j)
                sumXY += D[i][j];

        return (2.0 / (static_cast<double>(nX) * static_cast<double>(nY))) * sumXY
             - (2.0 / (static_cast<double>(nX) * static_cast<double>(nX))) * sumXX
             - (2.0 / (static_cast<double>(nY) * static_cast<double>(nY))) * sumYY;
    }

    // Energy statistic for an arbitrary permutation of pooled indices.
    // perm is a length-N array of pooled indices; the first nX entries are treated as X.
    inline double energy_statistic_from_pooled_permutation(
        const std::vector<std::vector<double>>& D,
        size_t nX,
        const std::vector<size_t>& perm
    ) {
        const size_t N = D.size();
        const size_t nY = N - nX;

        double sumXX = 0.0;
        for (size_t ii = 0; ii < nX; ++ii)
            for (size_t jj = ii + 1; jj < nX; ++jj)
                sumXX += D[perm[ii]][perm[jj]];

        double sumYY = 0.0;
        for (size_t ii = nX; ii < N; ++ii)
            for (size_t jj = ii + 1; jj < N; ++jj)
                sumYY += D[perm[ii]][perm[jj]];

        double sumXY = 0.0;
        for (size_t ii = 0; ii < nX; ++ii)
            for (size_t jj = nX; jj < N; ++jj)
                sumXY += D[perm[ii]][perm[jj]];

        return (2.0 / (static_cast<double>(nX) * static_cast<double>(nY))) * sumXY
             - (2.0 / (static_cast<double>(nX) * static_cast<double>(nX))) * sumXX
             - (2.0 / (static_cast<double>(nY) * static_cast<double>(nY))) * sumYY;
    }

    // Compute observed stat (identity split) and permutation stats, reusing pooled D.
    // Permutation representation matches CUDA: perm is a permutation of pooled indices,
    // and the first nX entries are treated as X, the rest as Y.
    inline PermutationStats energy_statistic_and_permutations_from_pooled(
        const std::vector<std::vector<double>>& D,
        size_t nX,
        int permutations,
        std::mt19937& gen
    ) {
        const size_t N = D.size();
        PermutationStats out;
        out.observed_statistic = energy_statistic_from_pooled(D, nX);
        out.permuted_statistics.resize(static_cast<size_t>(permutations));

        std::vector<size_t> perm(N);
        std::iota(perm.begin(), perm.end(), 0);
        for (int p = 0; p < permutations; ++p) {
            if (p != 0) {
                std::shuffle(perm.begin(), perm.end(), gen);
            }
            out.permuted_statistics[static_cast<size_t>(p)] =
                energy_statistic_from_pooled_permutation(D, nX, perm);
        }
        return out;
    }

    template <typename Container>
    PermutationStats energy_statistic_and_permutations(
        const Container& X,
        const Container& Y,
        int permutations,
        std::mt19937& gen
    ) {
        const size_t nX = X.size();
        auto D = compute_pooled_distance_matrix(X, Y);
        return energy_statistic_and_permutations_from_pooled(D, nX, permutations, gen);
    }

    template <typename Container>
    double energy_statistic(const Container& X, const Container& Y) {
        const size_t nX = X.size();
        auto D = compute_pooled_distance_matrix(X, Y);
        return energy_statistic_from_pooled(D, nX);
    }

    // Optimized p-value using precomputed pooled distance matrix
    template <typename Container>
    double calculate_p_value(Container X, Container Y, int permutations = 999) {
        const size_t nX = X.size();
        const size_t nY = Y.size();
        const size_t N = nX + nY;

        // Precompute pooled D once, then compute permutation stats from it.
        auto D = compute_pooled_distance_matrix(X, Y);

        std::random_device rd;
        std::mt19937 g(rd());
        auto stats = energy_statistic_and_permutations_from_pooled(D, nX, permutations, g);

        int count_greater = 0;
        for (int i = 0; i < permutations; ++i) {
            if (stats.permuted_statistics[static_cast<size_t>(i)] >= stats.observed_statistic)
                count_greater++;
        }
        return static_cast<double>(count_greater + 1) / (permutations + 1);
    }

}

#endif // ENERGY_TEST_CROSS_HPP

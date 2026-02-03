#ifndef ENERGY_TEST_CROSS_HPP
#define ENERGY_TEST_CROSS_HPP

#include <cmath>
#include <vector>
#include <algorithm>
#include <random>

namespace CPU_cross_dist {

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

    // Precompute distance matrices
    template <typename Container>
    void compute_distance_matrices(
        const Container& X, const Container& Y,
        std::vector<std::vector<double>>& DXX,
        std::vector<std::vector<double>>& DYY,
        std::vector<std::vector<double>>& DXY
    ) {
        size_t n = X.size();
        size_t m = Y.size();

        DXX.assign(n, std::vector<double>(n, 0.0));
        DYY.assign(m, std::vector<double>(m, 0.0));
        DXY.assign(n, std::vector<double>(m, 0.0));

        for (size_t i = 0; i < n; ++i)
            for (size_t j = i + 1; j < n; ++j) {
                double d = euclidean_dist(X[i], X[j]);
                DXX[i][j] = DXX[j][i] = d;
            }

        for (size_t i = 0; i < m; ++i)
            for (size_t j = i + 1; j < m; ++j) {
                double d = euclidean_dist(Y[i], Y[j]);
                DYY[i][j] = DYY[j][i] = d;
            }

        for (size_t i = 0; i < n; ++i)
            for (size_t j = 0; j < m; ++j)
                DXY[i][j] = euclidean_dist(X[i], Y[j]);
    }

    // Compute energy statistic using precomputed matrices
    inline double energy_statistic_from_matrices(
        const std::vector<std::vector<double>>& DXX,
        const std::vector<std::vector<double>>& DYY,
        const std::vector<std::vector<double>>& DXY
    ) {
        size_t n = DXX.size();
        size_t m = DYY.size();

        double sumXX = 0.0;
        for (size_t i = 0; i < n; ++i)
            for (size_t j = i + 1; j < n; ++j)
                sumXX += DXX[i][j];

        double sumYY = 0.0;
        for (size_t i = 0; i < m; ++i)
            for (size_t j = i + 1; j < m; ++j)
                sumYY += DYY[i][j];

        double sumXY = 0.0;
        for (size_t i = 0; i < n; ++i)
            for (size_t j = 0; j < m; ++j)
                sumXY += DXY[i][j];

        return (2.0 / (n * m)) * sumXY - (2.0 / (n * n)) * sumXX - (2.0 / (m * m)) * sumYY;
    }

    template <typename Container>
    double energy_statistic(const Container& X, const Container& Y) {
        std::vector<std::vector<double>> DXX, DYY, DXY;
        compute_distance_matrices(X, Y, DXX, DYY, DXY);
        return energy_statistic_from_matrices(DXX, DYY, DXY);
    }

    // Optimized p-value using precomputed pooled distance matrix
    template <typename Container>
    double calculate_p_value(Container X, Container Y, int permutations = 999) {
        size_t n = X.size();
        size_t m = Y.size();
        size_t N = n + m;

        // Pooled data
        Container pooled = X;
        pooled.insert(pooled.end(), Y.begin(), Y.end());

        // Precompute full pooled distance matrix
        std::vector<std::vector<double>> D(N, std::vector<double>(N, 0.0));
        for (size_t i = 0; i < N; ++i)
            for (size_t j = i + 1; j < N; ++j) {
                double d = euclidean_dist(pooled[i], pooled[j]);
                D[i][j] = D[j][i] = d;
            }

        // Helper to compute energy statistic from indices
        auto stat_from_indices = [&](const std::vector<size_t>& idxX) {
            std::vector<size_t> idxY;
            idxY.reserve(m);
            for (size_t i = 0; i < N; ++i) {
                if (std::find(idxX.begin(), idxX.end(), i) == idxX.end())
                    idxY.push_back(i);
            }

            double sumXX = 0.0;
            for (size_t i = 0; i < idxX.size(); ++i)
                for (size_t j = i + 1; j < idxX.size(); ++j)
                    sumXX += D[idxX[i]][idxX[j]];

            double sumYY = 0.0;
            for (size_t i = 0; i < idxY.size(); ++i)
                for (size_t j = i + 1; j < idxY.size(); ++j)
                    sumYY += D[idxY[i]][idxY[j]];

            double sumXY = 0.0;
            for (size_t i = 0; i < idxX.size(); ++i)
                for (size_t j = 0; j < idxY.size(); ++j)
                    sumXY += D[idxX[i]][idxY[j]];

            return (2.0 / (idxX.size() * idxY.size())) * sumXY
                 - (2.0 / (idxX.size() * idxX.size())) * sumXX
                 - (2.0 / (idxY.size() * idxY.size())) * sumYY;
        };

        // Observed statistic
        std::vector<size_t> idxX(n);
        std::iota(idxX.begin(), idxX.end(), 0);
        double observed_stat = stat_from_indices(idxX);

        // Permutations
        int count_greater = 0;
        std::vector<size_t> indices(N);
        std::iota(indices.begin(), indices.end(), 0);
        std::random_device rd;
        std::mt19937 g(rd());

        for (int p = 0; p < permutations; ++p) {
            std::shuffle(indices.begin(), indices.end(), g);
            std::vector<size_t> permX(indices.begin(), indices.begin() + n);
            double perm_stat = stat_from_indices(permX);
            if (perm_stat >= observed_stat) count_greater++;
        }

        return static_cast<double>(count_greater + 1) / (permutations + 1);
    }

}

#endif // ENERGY_TEST_CROSS_HPP

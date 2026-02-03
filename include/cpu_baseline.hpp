#ifndef ENERGY_TEST_HPP
#define ENERGY_TEST_HPP

#include <cmath>
#include <vector>
#include <algorithm>
#include <random>
#include <numeric>

namespace CPU {
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

    template <typename Container>
    double energy_statistic(const Container& X, const Container& Y) {
        double sumXY = 0, sumXX = 0, sumYY = 0;
        size_t n = X.size();
        size_t m = Y.size();

        // Cross-dist
        for (const auto& x : X) 
            for (const auto& y : Y) 
                sumXY += euclidean_dist(x, y);

        // Internal-dist-XX
        for (size_t i = 0; i < n; ++i)
            for (size_t j = i + 1; j < n; ++j) 
                sumXX += euclidean_dist(X[i], X[j]);

        // Internal-dist-YY
        for (size_t i = 0; i < m; ++i)
            for (size_t j = i + 1; j < m; ++j)
                sumYY += euclidean_dist(Y[i], Y[j]);

        // Note: Internal sums are multiplied by 2 because dist(a,b) == dist(b,a)
        // and dist(a,a) is 0. This is more efficient than N^2 loops.
        return (2.0 / (n * m)) * sumXY - (2.0 / (n * n)) * sumXX - (2.0 / (m * m)) * sumYY;
    }

    template <typename Container>
    double calculate_p_value(Container X, Container Y, int permutations = 999) {
        double observed_stat = energy_statistic(X, Y);
        
        // Prep pooled data
        Container pooled = X;
        pooled.insert(pooled.end(), Y.begin(), Y.end());
        
        int count_greater = 0;
        std::random_device rd;
        std::mt19937 g(rd());

        for (int i = 0; i < permutations; ++i) {
            std::shuffle(pooled.begin(), pooled.end(), g);
            
            // Partition the pooled data back into original sizes
            Container newX(pooled.begin(), pooled.begin() + X.size());
            Container newY(pooled.begin() + X.size(), pooled.end());
            
            if (energy_statistic(newX, newY) >= observed_stat) {
                count_greater++;
            }
        }

        return static_cast<double>(count_greater + 1) / (permutations + 1);
    }
}

#endif // ENERGY_TEST_HPP

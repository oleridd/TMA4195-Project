#ifndef OPM_GET_QUASI_IMPES_WEIGHTS_HEADER_INCLUDED
#define OPM_GET_QUASI_IMPES_WEIGHTS_HEADER_INCLUDED

#include <algorithm>
#include <cmath>

namespace Opm
{

namespace Details
{
    template <class DenseMatrix>
    DenseMatrix transposeDenseMatrix(const DenseMatrix& M)
    {
        DenseMatrix tmp;
        for (int i = 0; i < M.rows; ++i)
            for (int j = 0; j < M.cols; ++j)
                tmp[j][i] = M[i][j];

        return tmp;
    }
} // namespace Details

namespace Amg
{
    template <class Matrix, class Vector>
    void getQuasiImpesWeights(const Matrix& matrix, const int pressureVarIndex, Vector& weights, bool transpose = false)
    {
        using VectorBlockType = typename Vector::block_type;
        using MatrixBlockType = typename Matrix::block_type;
        const Matrix& A = matrix;
        VectorBlockType rhs(0.0);
        rhs[pressureVarIndex] = 1.0;
        const auto endi = A.end();
        for (auto i = A.begin(); i != endi; ++i) {
            const auto endj = (*i).end();
            MatrixBlockType diag_block(0.0);
            for (auto j = (*i).begin(); j != endj; ++j) {
                if (i.index() == j.index()) {
                    diag_block = (*j);
                    break;
                }
            }
            VectorBlockType bweights;
            if (transpose) {
                diag_block.solve(bweights, rhs);
            } else {
                auto diag_block_transpose = Opm::Details::transposeDenseMatrix(diag_block);
                diag_block_transpose.solve(bweights, rhs);
            }
            double abs_max = *std::max_element(
                bweights.begin(), bweights.end(), [](double a, double b) { return std::fabs(a) < std::fabs(b); });
            bweights /= std::fabs(abs_max);
            weights[i.index()] = bweights;
        }
        // return weights;
    }

    template <class Matrix, class Vector>
    Vector getQuasiImpesWeights(const Matrix& matrix, const int pressureVarIndex, bool transpose = false)
    {
        Vector weights(matrix.N());
        getQuasiImpesWeights(matrix, pressureVarIndex, weights, transpose);	
        return weights;
    }

} // namespace Amg

} // namespace Opm
#endif

#include <iostream>
#include <math.h>
#include <float.h>
#include <stdlib.h>
#include <stdio.h>
#include <cstring>
#include <string>
#include <fstream>
#include <time.h>
#include <typeinfo>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues> 
#include <limits>
#include<algorithm>
#include <vector>
#include <complex>

using namespace std;

vector<double> Hbeta(vector<double> D, double beta = 1.0)
{
    // P = np.exp(-D.copy() * beta)

    vector<double> P;

    for (int i = 0; i < D.size(); i++)
    {
        P.push_back(exp(-1 * D[i] * beta));
    }

    // sumP = sum(P)
    double sumP = 0.0;

    // Sum of matrix
    for (int i = 0; i < P.size(); i++)
    {
        sumP += P[i];
    }

    // np.sum(D * P)

    double sum_v;

    // Sum of matrix
    for (int i = 0; i < D.size(); i++)
    {
        sum_v += D[i] * P[i];
    }

    // H = np.log(sumP) + beta * np.sum(D * P) / sumP
    double H = log(sumP) + beta * sum_v / sumP;

    P.insert(P.end(), H);

    // Returns a matrix P, in the last element is H (vector)
    return P;
}

Eigen::MatrixXd pca(vector<vector<double> >X, int no_dims=50){
    // covirtiendo X (vector de vectores) a Matrix Eigen

    int rows = X.size();
    int columns = X[0].size();

    Eigen::MatrixXd x(rows, columns);

    for(int i = 0; i < rows; i++){
        for (int j = 0; j < columns; j++){
            x(i,j) = X[i][j];
        }
    }
    // np.mean(X,0)
    Eigen::MatrixXd x_mean(columns, 1);
    x_mean = x.colwise().mean();

    // np.tile(np.mean(X, 0), (n, 1))
    Eigen::MatrixXd x_tile(rows, columns);
    x_tile = x_mean.replicate(rows,1);

    // X = X - np.tile(np.mean(X, 0), (n, 1))
    Eigen::MatrixXd x_sub;
    x_sub = x - x_tile; // new x

    // X.T
    Eigen::MatrixXd x_transp;
    x_transp = x_sub.transpose();

    // np.dot(X.T, X)
    Eigen::MatrixXd x_dot;
    x_dot = x_transp * x_sub;

    // np.linalg.eig(np.dot(X.T, X))
    Eigen::EigenSolver<Eigen::MatrixXd> eigensolver(x_dot);
    Eigen::MatrixXcd eigen_vectors = eigensolver.eigenvectors();

    for (int i = 0; i < eigen_vectors.rows(); i++){
        for (int j = 0; j < eigen_vectors.cols(); j++){
            eigen_vectors(i,j) *= -1.0; 
        }
    }

    // M[:, 0:no_dims]
    Eigen::MatrixXcd M = eigen_vectors.block(0,0, columns, no_dims);

    // Y = np.dot(X, M[:, 0:no_dims])
    Eigen::MatrixXcd Y;
    Y = x_sub * M;

    Eigen::MatrixXd Y_real;
    Y_real = Y.real();

    return Y_real;
}

Eigen::MatrixXd x2p(vector<vector<double> >X, double tol=1e-5, double perplexity=30.0){
    // covirtiendo X (vector de vectores) a Matrix Eigen

    int rows = X.size();
    int columns = X[0].size();

    Eigen::MatrixXd x(rows, columns);

    for(int i = 0; i < rows; i++){
        for (int j = 0; j < columns; j++){
            x(i,j) = X[i][j];
        }
    }

    // np.square(X)
    Eigen::MatrixXd x_squared(rows, columns);
    for (int i = 0; i < rows; i++){
        for (int j = 0; j < columns; j++){
            x_squared(i,j) = pow(x(i,j), 2);
        }
    }

    // np.sum(np.square(X), 1)
    Eigen::MatrixXd x_sum;
    x_sum = x_squared.rowwise().sum().transpose();

    // // np.dot(X, X.T)
    Eigen::MatrixXd x_dot;
    x_dot = x * x.transpose();

    // // -2 * np.dot(X, X.T)
    Eigen::MatrixXd x_dot_mult;
    x_dot_mult = -2 * x_dot;

    // np.add(-2 * np.dot(X, X.T), sum_X)
    Eigen::MatrixXd np_add(rows,rows);
     for (int i = 0; i < rows; i++){
        for (int j = 0; j < rows; j++){
            np_add(i,j) = x_dot_mult(i,j) + x_sum(0,j);
        } 
    }

    // np.add(-2 * np.dot(X, X.T), sum_X).T
    Eigen::MatrixXd np_add_transp;
    np_add_transp = np_add.transpose();

    // D = np.add(np.add(-2 * np.dot(X, X.T), sum_X).T, sum_X)
    Eigen::MatrixXd D(rows, rows);
     for (int i = 0; i < rows; i++){
        for (int j = 0; j < rows; j++){
            D(i,j) = np_add_transp(i,j) + x_sum(0,j);
        } 
    }

    // P = np.zeros((n, n))
    Eigen::MatrixXd P(rows, rows);
    P = P.setZero();

    // beta = np.ones((n, 1))
    Eigen::MatrixXd beta(rows, 1);
    beta = beta.setOnes();

    // logU = np.log(perplexity)
    double logU = log(perplexity);

    // for i in range(n):
    for(int i = 0; i < rows; i++){

        Eigen::MatrixXi np_r_1(1,i);
        Eigen::MatrixXi np_r_2(1, (rows-1)-i);

        // betamin = -np.inf
        double betamin = -std::numeric_limits<double>::infinity(); 

        // betamax = np.inf
        double betamax = std::numeric_limits<double>::infinity(); 

        // np.r_[0:i]
        for (int j = 0; j < i; j++){
            np_r_1(0,j) = j;
        }

        // np.r_[i + 1:n]
        int k = i;
        for(int j = 0; j < (rows-1)-i; j++){
            np_r_2(0, j) = k+1;
            k++;
        }

        // np.concatenate((np.r_[0:i], np.r_[i+1:n]))
        Eigen::MatrixXi concatenate(np_r_1.rows(), np_r_1.cols()+np_r_2.cols());
        concatenate << np_r_1, np_r_2;

        // Di = D[i, np.concatenate((np.r_[0:i], np.r_[i+1:n]))]
        int m = i;
        Eigen::MatrixXd Di(1,rows-1);
        for(int i = 0; i < rows-1; i++){
            Di(0,i) = D(m, concatenate(0,i));
        }
        
        // (H, thisP) = Hbeta(Di, beta[i])
        vector<double> Di_v;
        for(int i = 0; i < rows-1; i++){
           Di_v.push_back(Di(0,i));
        }

        vector<double> thisP;
        thisP = Hbeta(Di_v, beta(i,0));

        double H = thisP.back();

        thisP.pop_back();

        // Hdiff = H - logU
        double Hdiff = H - logU;

        // tries = 0
        int tries = 0;

        // while np.abs(Hdiff) > tol and tries < 50:
        while((abs(Hdiff) > tol) && (tries < 50)){

            if (Hdiff > 0){
                // betamin = beta[i].copy()
                betamin = beta(i,0);

                // if betamax == np.inf or betamax == -np.inf:
                if ((betamax == std::numeric_limits<double>::infinity()) || (betamax == -std::numeric_limits<double>::infinity())){
                    // beta[i] = beta[i] * 2.
                    beta(i,0) = beta(i,0) * 2.0;

                }else{
                    // beta[i] = (beta[i] + betamax) / 2.
                    beta(i,0) = (beta(i,0) + betamax) / 2.0;
                }
            
            }else{
                // betamax = beta[i].copy()
                betamax = beta(i,0);

                // if betamin == np.inf or betamin == -np.inf:
                if ((betamin == std::numeric_limits<double>::infinity()) || (betamin == -std::numeric_limits<double>::infinity())){
                    // beta[i] = beta[i] / 2.
                    beta(i,0) = beta(i,0) / 2.0;
                }else{
                    // beta[i] = (beta[i] + betamin) / 2.
                    beta(i,0) = (beta(i,0) + betamin) / 2.0;
                }
            }

            // (H, thisP) = Hbeta(Di, beta[i])
            thisP = Hbeta(Di_v, beta(i,0));
            H = thisP.back();
            thisP.pop_back();

            // Hdiff = H - logU
            Hdiff = H - logU;

            // tries += 1
            tries++;
        }

        // np.r_[0:i]
        for (int j = 0; j < i; j++){
            np_r_1(0,j) = j;
        }

        // np.r_[i + 1:n]
        k = i;
        for(int j = 0; j < (rows-1)-i; j++){
            np_r_2(0, j) = k+1;
            k++;
        }

        // np.concatenate((np.r_[0:i], np.r_[i+1:n]))
        concatenate << np_r_1, np_r_2;

        // P[i, np.concatenate((np.r_[0:i], np.r_[i+1:n]))] = thisP
        for (int j = 0; j <  concatenate.cols(); j++){
            P(i, concatenate(0,j)) = thisP[j];
        }

        // if(i == 100){
        //     cout<<P<<"\n";
        // }

    } 

    // np.sqrt(1/beta)
    Eigen::MatrixXd sqrt_beta(beta.rows(), beta.cols());
    for (int i = 0; i < sqrt_beta.rows(); i++){
        for (int j = 0; j < sqrt_beta.cols(); j++){
            sqrt_beta(i,j) = sqrt((1.0 / beta(i,j)));
        }
    }

    // np.mean(np.sqrt(1 / beta))
    double mean_sqrt_beta;
    mean_sqrt_beta = sqrt_beta.mean();

    // print("Mean value of sigma: %f" % np.mean(np.sqrt(1 / beta)))
    cout<<"Mean value of sigma: "<< mean_sqrt_beta<<"\n";

    // PARA TSNE_2
    return P;
}

Eigen::MatrixXd tsne(vector<vector<double> > X_init, vector<vector<double> > rand_num, int no_dims = 2, int initial_dims = 50, double perplexity = 30.0)
{

    // X = pca(X, initial_dims).real

    Eigen::MatrixXd X = pca(X_init, initial_dims);

    int n = X.rows();
    int d = X.cols();
    int max_iter = 100;
    double initial_momentum = 0.7;
    double final_momentum = 0.9;
    int eta = 500;
    double min_gain = 0.1;

    // Y = np.random.randn(n, no_dims)
    Eigen::MatrixXd Y(n, no_dims);
    for (int i = 0; i < Y.rows(); i++){
        for (int j = 0; j < Y.cols(); j++){
            Y(i,j) = rand_num[i][j];
        }
    }
    
    // for (int i = 0; i < Y.rows(); i++)
    // {
    //     for (int j = 0; j < Y.cols(); j++)
    //     {
    //         Y(i, j) = static_cast<double>(rand()) / static_cast<double>(RAND_MAX);
    //     }
    // }

    Eigen::MatrixXd dY(n, no_dims);
    Eigen::MatrixXd iY(n, no_dims);
    Eigen::MatrixXd gains(n, no_dims);

    // Compute P-values
    vector<vector<double> > X_2(X.rows(), vector<double>(X.cols(), 0));
    for (int i = 0; i < X.rows(); i++){
        for (int j = 0; j < X.cols(); j++){
            X_2[i][j] = X(i,j);
        }
    }
    
    Eigen::MatrixXd P = x2p(X_2, 1e-5, perplexity);

    // P = P + np.transpose(P)
    Eigen::MatrixXd P_t = P.transpose();

    P = P + P_t;

    // P = P / np.sum(P)
    P = P / P.sum();

    // P = P * 4.
    P = P * 4.0;

    // P = np.maximum(P, 1e-12)
    for (int f = 0; f < P.rows(); f++)
    {
        for (int g = 0; g < P.cols(); g++)
        {
            if (P(f, g) < 1e-12)
            {
                P(f, g) = 1e-12;
            }
        }
    }

    // for iter in range(max_iter):
    for (int iter = 0; iter < max_iter; iter++)
    {
        // sum_Y = np.sum(np.square(Y), 1)
        Eigen::MatrixXd sum_Y(Y.rows(), 1);
        Eigen::MatrixXd square(Y.rows(), Y.cols());

        // sum_Y = np.sum(np.square(Y), 1)
        for (int f = 0; f < Y.rows(); f++)
        {
            for (int g = 0; g < Y.cols(); g++)
            {
                square(f, g) = pow(Y(f, g), 2);
            }
        }

        sum_Y = Y.rowwise().sum();

        // num = -2. * np.dot(Y, Y.T)
        Eigen::MatrixXd num(Y.rows(), Y.rows());
        Eigen::MatrixXd Y_t(Y.rows(), Y.rows());
        
        Y_t = Y.transpose();

        num = -2.0 * Y * Y_t;

        // np.add(num, sum_Y).T
        Eigen::MatrixXd add(Y.rows(), Y.rows());

        for (int f = 0; f < Y.rows(); f++){
            for (int g = 0; g < Y.cols(); g++){
                add(f,g) =  num(f,g) + sum_Y(g,0);
            }
        }

        // add.transposeInPlace();

        // np.add(np.add(num, sum_Y).T, sum_Y)
        Eigen::MatrixXd add_2(add.rows(), add.cols());

        for (int f = 0; f < add.rows(); f++){
            for (int g = 0; g < add.cols(); g++){
                add_2(g, f) =  add(f,g) + sum_Y(g,0);
            }
        }

        // num = 1. / (1. + np.add(np.add(num, sum_Y).T, sum_Y))
        for (int f = 0; f <  num.rows(); f++){
            for (int g = 0; g < num.cols(); g++){
                num(f, g) =  1.0 / (1.0 + add_2(f, g));
            }
        }

        // num[range(n), range(n)] = 0.
        for (int i = 0; i < num.rows(); i++){
            num(i, i) = 0.0;
        }

        // Q = num / np.sum(num)
        Eigen::MatrixXd Q(num.rows(), num.cols());
        Q = num / num.sum();

       // Q = np.maximum(Q, 1e-12)

        for (int i = 0; i < Q.rows(); i++){
            for (int j = 0; j < Q.cols(); j++){
                if (Q(i, j) < 1e-12){
                    Q(i, j) = 1e-12;
                }
            }
        }

        // Compute Gradient

        // PQ = P - Q
        Eigen::MatrixXd PQ(P.rows(), P.cols());
        PQ = P - Q;
        
        // for i in range(n):
        for (int i = 0; i < n; i++){
            // dY[i, :] = np.sum(np.tile(PQ[:, i] * num[:, i], (no_dims, 1)).T * (Y[i, :] - Y), 0)

            // PQ[:, i] * num[:, i]
            Eigen::MatrixXd som(1, PQ.cols());

            for(int j = 0; j < som.cols(); j++){
                som(0, j) = PQ(j, i) * num(j, i);
            }

            // np.tile(som, (no_dims, 1)) = Shape(2, 2500)
            Eigen::MatrixXd som_tim(no_dims, som.rows());
            // som_tim = som.replicate(no_dims, 2);
            som_tim = som.replicate(no_dims, 1);

            // som_tim.T (2500, 2)
            som_tim.transposeInPlace();

            //(Y[i, :] - Y)
            Eigen::MatrixXd rest(Y.rows(), Y.cols());
            for (int j = 0; j < Y.rows(); j++){
                for (int r = 0; r < Y.cols(); r++){
                    rest(j, r) = Y(i, r) - Y(j, r);
                }
            }

            // som_tim * rest
            for (int j = 0; j < som_tim.rows(); j++){
                for (int r = 0; r < som_tim.cols(); r++){
                    som_tim(j, r) *= rest(j, r);
                }
            }            

            for (int j = 0; j < dY.cols(); j++){
                dY(i, j) = som_tim.colwise().sum()(0,j); 
            }
            
        }

        
        // Perform the update

        double momentum = 0.0;

        /* if iter < 20:
        //     momentum = initial_momentum
        // else:
        //     momentum = final_momentum */

        if (iter < 20){
            momentum = initial_momentum;
        }else{
            momentum = final_momentum;
        }

        // gains = (gains + 0.2) * ((dY > 0.) != (iY > 0.)) + (gains * 0.8) * ((dY > 0.) == (iY > 0.))
        for(int j = 0; j < gains.rows(); j++){
            for(int r = 0; r < gains.cols(); r++){
                double gain = gains(j,r);
                if ((dY(j, r) > 0.0) != (iY(j, r) > 0.0)){
                    gain += 0.2;
                }else{
                    gain *= 0.8;
                }

                gains(j,r) = gain;
            }
        }

        // gains[gains < min_gain] = min_gain
        for (int j = 0; j < gains.rows(); j++){
            for (int r = 0; r < gains.cols(); r++){
                if (gains(j, r) < min_gain){
                    gains(j, r) = min_gain;
                }
            }
        }

        // iY = momentum * iY - eta * (gains * dY)
        // Y = Y + iY
        for (int j = 0; j < gains.rows(); j++){
            for (int r = 0; r < gains.cols(); r++){
                iY(j, r) = momentum * iY(j, r) - eta * (gains(j, r) * dY(j, r));
                Y(j, r) += iY(j, r);
            }
        }

        // Y = Y - np.tile(np.mean(Y, 0), (n, 1))
        // np.mean(Y, 0)
        Eigen::MatrixXd mean(1, Y.cols());
        mean = Y.colwise().mean();

        // Y = Y - np.tile(mean, (n, 1))
        Y = Y - mean.replicate(n, 1);

        // Compute current value of cost function
        // if (iter + 1) % 10 == 0:
        Eigen::MatrixXd mult(P.rows(), P.cols());
        if ((iter + 1) % 5 == 0){
            // C = np.sum(P * np.log(P / Q))
            for (int j = 0; j < P.rows(); j++){
                for (int r = 0; r < P.cols(); r++){
                    mult(j, r) = P(j, r) * log(P(j, r) / Q(j, r));
                }
            }

            cout<<"Iteration " << iter + 1 << ": error is "<< mult.sum() <<"\n";
        }

        // Stop lying about P-values
        /*if iter == 100:
            P = P / 4.*/

        if (iter == 100){
            P = P / 4.0;
        }

    }
    return Y;
}


int main(){

    // LOAD DATA

    int rows = 2500;
    int columns = 784;
    int i, j;

    vector<vector<double> > m(2500, vector<double>(784, 0));

    ifstream infile("./mnist2500_X.txt");

    if(!infile){
        cout << "Cannot open file.\n";
        return 0;
    }

    for (i = 0; i < rows; i++){
        for (j = 0; j < columns; j++){
            infile >> m[i][j];
        }
    }

    infile.close();

    vector<vector<double> > r(2500, vector<double>(2, 0));

    ifstream infile2("./readme.txt");

    if (!infile2){
        cout << "Cannot open file.\n";
        return 0;
    }

    for(i = 0; i < 2500; i++){
        for(j = 0; j < 2; j++){
            infile2 >> r[i][j];
        }
    }

    infile2.close();

    Eigen::MatrixXd Y(rows, 2);
    Y = tsne(m, r, 2, 50, 20.0);

    ofstream fw("./result.txt", std::ofstream::out);

    if (fw.is_open()){
        for(int i = 0; i<Y.rows(); i++){
            fw << Y(i,0) << " " << Y(i,1) <<"\n";
        }

        fw.close();
    }
}
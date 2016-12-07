package com.company;

import org.apache.commons.math3.linear.LUDecomposition;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.SingularMatrixException;

/**
 *
 * @author Xin
 */

/* Reference:
      Maximum Likelihood Estimation of Logistic Regression Models: Theory and Implementation
      Scott A. Czepiel∗
*/
public class MultinomialLR {
    public int J; // number of discrete values of y
    public int N; //number of populations
    public int K; //number of columns in x;
    public double n[];// = new double[N]; //population counts
    public double y[][];// = new double[N][J-1]; //dv counts
    public double x[][];// = new double[N][K+1]; //design matrix
    double pr[][]; // probabilities
    double beta[];//parameters
    double xrange[];//range of x
    double stdError[];
    double Zscore[];

    public MultinomialLR(double[] n, double[][] y, double[][] x, int N, int K, int J){
        this.n = n;
        this.y = y;
        this.x = x;
        this.N = N;
        this.K = K;
        this.J = J;
    }
    public void fit(){

        int i,j,k;
        int max_iter = 15;
        double eps = 1e-8;
        int iter = 0;
        boolean converged = false;
        double beta_old[] = new double[(K + 1) * (J - 1)];
        double beta_inf[] = new double[(K + 1) * (J - 1)];
        double xtwx[][] = new double[(K + 1) * (J - 1)][(K + 1) * (J - 1)];
        double loglike = 0;
        double loglike_old = 0;
        pr= new double[N][J - 1];
        beta = new double[(K + 1) * (J - 1)];
        xrange = new double[(K + 1) * (J - 1)];
        stdError = new double[(K + 1) * (J - 1)];
        //initialize parameters to zero
        for(k = 0; k<(K+1)*(J-1);k++){
            beta[k]=0;
            beta_inf[k]=0;
            for(j = 0; j<(K+1)*(J-1);j++){
                xtwx[k][j]=0;
            }
        }
        //main loop
        while(iter < max_iter && !converged){
            for(k = 0; k<(K+1)*(J-1);k++){
                beta_old[k]=beta[k];
            }
            
            //run one iteration of newton-raphson
            loglike_old = loglike;
            loglike = newton_raphson(J,N,K,n,y,pr,x,beta,xtwx);
            
            for (i = 0; i < (K + 1) * (J - 1); i++) {

                
                    stdError[i] = Math.sqrt(xtwx[i][i]);
                
            }
            
            //test for decreasing likelihood ,there is a problem need to be fixed
            if(loglike < loglike_old && iter >0){
                System.out.print("\nLikelihood start decreasing.\n");
                break;
            }
            //test for infinite parameters
            for(k = 0; k<(K+1)*(J-1);k++){
                if(beta_inf[k]!=0){
                    beta[k]=beta_inf[k];
                }else{
                    if(Math.abs(beta[k])>(5/xrange[k])
                            && Math.sqrt(xtwx[k][k])>=(3*Math.abs(beta[k]))){
                        beta_inf[k]=beta[k];
                    }
                }
            }
            // test for convergence
            converged = true;
            for(k = 0; k<(K+1)*(J-1);k++){
                if(Math.abs(beta[k]-beta_old[k])>eps*Math.abs(beta_old[k])){
                    converged = false;
                    System.out.print("GG");
                    break;
                }
            }
            iter++;
            System.out.println("..."+iter);
        }//end of main loop

    }

    public double newton_raphson(int J, int N, int K, double n[],
                                 double y[][], double pr[][], double x[][],double beta[],double xtwx[][]) {

        int i, j, jj, jprime, k, kk, kprime;
        double beta_tmp[] = new double[(K+1)*(J-1)];
        double xtwx_tmp[][] = new double[(K+1)*(J-1)][(K+1)*(J-1)];
        double xtwx_inverse[][];
        double loglike = 0;
        double denom;
        double numer[] = new double[J - 1]; // length J-1
        double tmp1, tmp2, w1, w2;
        logGamma logGamma = new logGamma();
        // main loop for each row in the design matrix */
        for (i = 0; i < N; i++) {
            // matrix multiply one row of x * beta
            denom = 1;
            for (j = 0; j < J - 1; j++) {
                tmp1 = 0;
                for (k = 0; k < K + 1; k++) {
                    tmp1 += x[i][k] * beta[j * (K + 1) + k];
                    
                }
                numer[j] = Math.exp(tmp1);
                denom += numer[j];
            }
            // calculate predicted probabilities
            for (j = 0; j < J - 1; j++) {
                pr[i][j] = numer[j] / denom;
            }
            // add log likelihood for current row
            loglike += logGamma.gamma(n[i] + 1);
            for (j = 0, tmp1 = 0, tmp2 = 0; j < J - 1; j++) {
                tmp1 += y[i][j];
                tmp2 += pr[i][j];
                loglike = loglike - logGamma.gamma(y[i][j]+1) +
                        y[i][j] * Math.log(pr[i][j]);
            }
            // Jth category
            loglike = loglike - logGamma.gamma(n[i]-tmp1+1) +
                    (n[i]-tmp1) * Math.log(1-tmp2);

            // add first and second derivatives
            for (j = 0, jj = 0; j < J - 1; j++) {
                tmp1 = y[i][j] - n[i] * pr[i][j];
                w1 = n[i] * pr[i][j] * (1 - pr[i][j]);
                for (k = 0; k < K + 1; k++) {
                    //ystem.out.println("sad"+jj);
                    beta_tmp[jj] += tmp1 * x[i][k];
                    
                    kk = jj - 1;
                    for (kprime = k; kprime < K + 1; kprime++) {
                        kk++;
                        xtwx_tmp[jj][kk] += w1 * x[i][k] * x[i][kprime];
                        xtwx_tmp[kk][jj] = xtwx_tmp[jj][kk];
                    }
                    for (jprime = j + 1; jprime< J - 1; jprime++) {
                        w2 = -n[i] * pr[i][j] * pr[i][jprime];
                        for (kprime = 0; kprime < K + 1; kprime++) {
                            kk++;
                            xtwx_tmp[jj][kk] +=
                                    w2 * x[i][k] * x[i][kprime];
                            xtwx_tmp[kk][jj] = xtwx_tmp[jj][kk];
                        }
                    }
                    jj++;
                }
            }
        } // end loop for each row in design matrix
        
        // compute xtwx matrix's inverse
        RealMatrix m = MatrixUtils.createRealMatrix(xtwx_tmp);
        try{
            
            RealMatrix mInverse = new LUDecomposition(m).getSolver().getInverse();
            xtwx_inverse = mInverse.getData();



            // compute xtwx * beta(0) + x(y-mu)
            for (i = 0; i < (K + 1) * (J - 1); i++) {
                tmp1 = 0;
                for (j = 0; j < (K + 1) * (J - 1); j++) {
                    tmp1 += xtwx_tmp[i][j] * beta[j]; 
                }
                beta_tmp[i] += tmp1;
                //System.out.println("danhaoteng"+beta_tmp[i]);
            }
            // solve for new betas
            for (i = 0; i < (K + 1) * (J - 1); i++) {
                tmp1 = 0;
                for (j = 0;j < (K + 1) * (J - 1); j++) {
                    tmp1 += xtwx_inverse[i][j] * beta_tmp[j];
                    //System.out.println("danhaotengaaaa"+xtwx_tmp[i][j]);
                }
                beta[i] = tmp1;

            }

            for (i = 0; i < (K + 1) * (J - 1); i++) {
                for (j = 0; j < (K + 1) * (J - 1); j++) {
                   xtwx[i][j] = xtwx_inverse[i][j];
                }
            }

            System.out.println("-.-"+loglike);
        }catch(SingularMatrixException e){
            
        }
        return loglike;
    }

    public double[] getCoefficient(){
        for(int n = 0; n < (K+1)*(J-1); n++){
            
            System.out.println(beta[n]);
            
        }
        return beta;
    }
    
    public double[] getSDError(){
        return stdError;
    }
    
    public double[] getZscore(){
        Zscore = new double[(K+1)*(J-1)];
        for(int n = 0; n < (K+1)*(J-1); n++){
            
            Zscore[n]=beta[n]/stdError[n];
            
        }
        return Zscore;
    }
}


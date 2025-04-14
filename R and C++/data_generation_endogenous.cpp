#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]

using namespace Rcpp;

/*
 * Compute Probability From Logistic Function
 * 
 * This function compute the probability using the standard logistic function 
 * given a value. 
 */
// [[Rcpp::export]]
double logistic_function(double x) {
  return exp(x) / (1 + exp(x));
}

/*
 * Compute Transition Probability Matrix 
 */
// [[Rcpp::export]]
NumericMatrix compute_QTrans_prob_logistic(
    double beta10_0, // Parameters transition State 0 --> State 1
    double beta10_1,
    double beta10_2,
    double beta01_0, // Parameters transition state 1 --> state 0
    double beta01_1,
    double beta01_2,
    double Y, 
    double L
  ) {
  
  // Compute probabilities using logistic function
  double p10 = logistic_function(beta10_0 + beta10_1 * Y + beta10_2 * L);
  double p01 = logistic_function(beta01_0 + beta01_1 * Y + beta01_2 * L);
  
  // Create 2x2 transition probability matrix
  NumericMatrix QTransProb(2, 2);
  QTransProb(0, 0) = 1 - p10;
  QTransProb(0, 1) = p01;
  QTransProb(1, 0) = p10;
  QTransProb(1, 1) = 1 - p01;
  
  return QTransProb;
}

/*
 * Generate DT-Trivariate Process
 * 
 * This function generates a discrete time (DT) trivariate process with a
 * time-varying exposure X, time-varying confounder L, and time-varying outcome 
 * Y. Data are preprocessed, which includes mean-aggregation and labeling 
 * everyth data point. 
 */
// [[Rcpp::export]]
DataFrame generate_data_V3(
  int sample_size, 
  int n_time_points, // Number of one-unit time points
  int m, // Number of time-units per frame
  int everyth_frame,
  int burnin,
  arma::mat phi, 
  arma::mat beta0, 
  arma::mat sigma2Eta, // Residual variance latent outcome process
  arma::mat sigma2L,
  arma::vec lambda, // Factor loading
  double tau, // Intercept observed variable
  double thetaY, // Measurement error observed outcome 
  double beta10_0, 
  double beta10_1,
  double beta10_2,
  double beta01_0,
  double beta01_1,
  double beta01_2,
  double gamma, // Effect time-varying covariate -> outcome
  double alpha, // Autoregression time-varying covariate
  bool preprocess
) {
  
  // Parameters: Simulation
  int n_time_total = n_time_points + burnin;
  int tt_burnin = burnin - 1; // Index end of burnin
  
  List df(sample_size);
  
  // Loop over units
  for (int i = 0; i < sample_size; ++i) { 
    
    // Initialize unit-vectors
    arma::vec eta = arma::zeros(n_time_total); // Latent outcome process
    arma::vec Yi = arma::zeros(n_time_total); // Observed outcome 
    IntegerVector Xi(n_time_total); // Exposure process
    arma::vec Li = arma::zeros(n_time_total); // Time-varying covariate

    // Initial values
    Yi[0] = eta[0] = R::rnorm(0, 1);
    Xi[0] = 0;
    Li[0] = R::rnorm(0, 1); 
    
    // Loop over time points per unit
    for (int t = 1; t < n_time_total; ++t) { 
      
      // Error term time-varying covariate
      arma::vec epsL = arma::randn(1) * sqrt(sigma2L);
      
      // Latent time-varying covariate process
      Li[t] = as_scalar(alpha * Li[t - 1] + epsL);
      
      // Compute transition probabilities
      NumericMatrix QProb = compute_QTrans_prob_logistic(
        beta10_0, beta10_1, beta10_2,
        beta01_0, beta01_1, beta01_2, 
        Yi[t - 1], Li[t]
      );
      
      // Generate new X[u]
      if(Xi[t - 1] == 0) {
        Xi[t] = R::rbinom(1, QProb(1, 0)); 
      } else {
        Xi[t] = R::rbinom(1, QProb(1, 1)); 
      }
      
      // Error term latent outcome process
      arma::vec epsEta = arma::randn(1) * sqrt(sigma2Eta);
      
      // Latent process(es)
      eta[t] = as_scalar(
        phi * eta[t - 1] + // Autoregression
          gamma * Li[t] + // Effect covariate
          beta0 * Xi[t] + // Effect exposure
          epsEta // Residual
        );
      
      // Measurement error
      double epsY = R::rnorm(0, thetaY);
      
      // Measurement model outcome
      Yi[t] = as_scalar(lambda.t() * eta[t]) + tau + epsY; 
    }
    
    // Combine df for unit i
    DataFrame dfi = DataFrame::create(
      Named("X") = Xi, 
      Named("Y") = Yi, 
      Named("L") = Li
    );
    df[i] = dfi;
  }
  
  // Initialize unit-combined vectors
  std::vector<double> X, Y, L; 
  std::vector<int> unit, frame;
  
  if (!preprocess) {
    
    // Initialize unit-combined vectors for data without preprocessing
    std::vector<double> X_aggr, Y_aggr, L_aggr; 
    std::vector<int> everyth;
    
    for (int i = 0; i < sample_size; ++i) { // Loop over units
      
      // Select unit-data
      DataFrame dfi = df[i]; 
      IntegerVector Xi = dfi["X"];
      NumericVector Yi = dfi["Y"];
      NumericVector Li = dfi["L"]; 
      
      // Vectors for storing data per frame
      double X_temp = 0;
      double Y_temp = 0;
      double L_temp = 0;
      
      // Frame number
      int tt_frame = 1; 
      
      for (int t = tt_burnin; t < (n_time_total - 1); ++t) { // Start from tt_burnin to remove burnin-period
        
        // Cummulative up to t
        X_temp += Xi[t];
        Y_temp += Yi[t];
        L_temp += Li[t];
        
        // Add mean-aggregate score up to t to unit-combined vectors
        X_aggr.push_back(X_temp / m);
        Y_aggr.push_back(Y_temp / m);
        L_aggr.push_back(L_temp / m);
        
        // Add time-point specific score to unit-combined vectors
        X.push_back(Xi[t]);
        Y.push_back(Yi[t]);
        L.push_back(Li[t]);
        
        // Add unit and frame information
        unit.push_back(i + 1);
        frame.push_back(tt_frame);
        if (tt_frame % everyth_frame == 0 && t % m == 0) { // Label systematically
          everyth.push_back(1);
        } 
        else {
          everyth.push_back(0);
        }
        
        if (t % m == 0) { // If frame is complete:
          
          // Reset cummulative up to t
          X_temp = 0;
          Y_temp = 0;
          L_temp = 0;
          
          tt_frame++; // Increase frame number
        } 
      }
    }
    
    // Rename columns
    DataFrame out = DataFrame::create(
      Named("X") = X,
      Named("L") = L,
      Named("Y") = Y,
      Named("X_aggr") = X_aggr,
      Named("L_aggr") = L_aggr,
      Named("Y_aggr") = Y_aggr,
      Named("frame") = frame,
      Named("unit") = unit,
      Named("everyth") = everyth
    );
    
    return out;
  } 
  else {
    for (int i = 0; i < sample_size; ++i) { // Loop over units
      
      // Select unit-data
      DataFrame dfi = df[i]; 
      IntegerVector Xi = dfi["X"];
      NumericVector Yi = dfi["Y"];
      NumericVector Li = dfi["L"]; 
      
      // Vectors for storing data per frame
      double X_temp = 0;
      double Y_temp = 0;
      double L_temp = 0;
      
      // Frame number
      int tt_frame = 1; 
      int tt_k = 1;
      
      for (int t = tt_burnin; t < (n_time_total - 1); ++t) { // Start from tt_burnin to remove burnin-period
        
        // Cummulative up to t
        X_temp += Xi[t];
        Y_temp += Yi[t];
        L_temp += Li[t];
        
        if(t % m == 0) { // If frame is complete and systematic sample
          
          if (tt_frame % everyth_frame == 0) { // If at everyth frame:
            
            // Add mean-aggregate score to unit-combined vectors
            X.push_back(X_temp / m);
            Y.push_back(Y_temp / m);
            L.push_back(L_temp / m);
            
            // Add unit and frame information
            unit.push_back(i + 1);
            frame.push_back(tt_k);
            
            tt_k++; // Increase k 
          }
          
          // Reset cummulative up to t
          X_temp = 0;
          Y_temp = 0;
          L_temp = 0;
          
          tt_frame++;
        } 
      }
    }

    // Rename columns
    DataFrame out = DataFrame::create(
      Named("X_aggr") = X,
      Named("L_aggr") = L,
      Named("Y_aggr") = Y,
      Named("frame") = frame,
      Named("unit") = unit
    );
    
    return out;
  }
}



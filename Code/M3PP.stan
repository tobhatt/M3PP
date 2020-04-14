//Markov modulated Marked Point Process (M3PP)
//Author: Tobias Hatt
//Date: 14.04.2020

functions{
//Transition probability
real transition_prob(int s_from, int s, int m, int d, matrix beta_trans, matrix alpha_trans, real[] delta_times){
    real duration;
    
    if(s_from == s){
     return negative_infinity();
    }    
    else{
    duration = sum(delta_times[(m-d+1):m]);

    
    return duration * beta_trans[s_from, s] + alpha_trans[s_from, s]; 
   }
}
//loglikelihood of arrival times (point process)
real compute_log_lik_HP(vector Time, int n, real time_window, real lengthscaleT, real mu, real a){
    //Time: vector observation times - these must be sorted from smallest to largest!
    //n: number of observation times in interval
    //time_window: time when interval is left
    //lengthscaleT: beta coefficient in exponential function
    //mu: base intensity
    //a: \frac{\alpha}{\beta}
    
    matrix[n,n] timeD;
    vector[n] timeD2;
    vector[n] ll = rep_vector(mu, n);
      
    for(i in 1:n) {
      for(j in 1:n) {
        timeD[i,j] = -(Time[i] - Time[j]);
      }
      timeD2[i] = -(time_window - Time[i]); 
    }
    
    for(i in 2:n) { 
      ll[i] += sum(a * lengthscaleT * exp(timeD[i, 1:(i-1)] * lengthscaleT));
    }
      
    return (sum(log(ll)) - time_window * mu + a * (sum(exp(timeD2*lengthscaleT)) - n));
}
//loglikelihood of observations in an interval (lieklihood of the marks)
real compute_log_lik_obs(int[] obs, real[] delta_time, row_vector phi_0, real[,] phi_1){
    int n = num_elements(obs);
    int P = num_elements(phi_0);

    vector[n] ll;
    
    ll[1] = phi_0[obs[1]] - log_sum_exp(phi_0);
    
    for(i in 2:n){
     ll[i] = phi_1[obs[i-1], obs[i]] - log_sum_exp(phi_1[obs[i-1]]);
    }    
    return sum(ll);
}
//total loglikelihood
real compute_log_lik(vector Time , int M , int D, int S , int L, vector l_pi , real[,,] log_density_duration, 
                       real[,,] time_log_lik, real[,,] obs_log_lik, real[,,,] l_tpm){
    
    //m: period
    //d: number of observations with \tau_n <= t_m < \tau_{n+1}
    //M: censoring time/period
    //L: Purchase or exit sequence
    
    //Initialisation
    real l_alpha[M,S,D];
    real l_acc_1[D];
    real l_alpha_star[S-1];
    int pointer;
    real log_lik[S];
    int d_prime;

    //Initialise l_alpha    
    l_alpha = rep_array(negative_infinity(), M, S, D);

    //m=1
    for(s in 1:S){
     l_alpha[1, s, 1] = l_pi[s] + log_density_duration[s, 1, 1] + time_log_lik[s, 1, 1] + obs_log_lik[s, 1, 1];       
    }
      
    // 2<=m<=D
    for(m in 2:min(D,M)){
     for(s in 1:S){     
      for(d in 1:(m-1)){
       pointer = 1;
       d_prime = min(D, m-d);
       for(i in 1:S){
        if(i != s){    
         l_alpha_star[pointer] = log_sum_exp(to_vector(l_alpha[m-d, i, 1:d_prime]) + to_vector(l_tpm[i, s, m-d, 1:d_prime]));
         pointer += 1;
        }
       }  
       l_alpha[m, s, d] = log_sum_exp(l_alpha_star) + log_density_duration[s, m-d+1, d] + time_log_lik[s, m-d+1, m] + obs_log_lik[s, m-d+1, m]; 
      }      
      l_alpha[m, s, m] = l_pi[s] + log_density_duration[s, 1, m] + time_log_lik[s, 1, m] + obs_log_lik[s, 1, m]; 
     }
    }    
    // D+1<=m<=M-1
    for(m in (D+1):M){  
     for(s in 1:S){                           
      for(d in 1:D){
       pointer = 1;
       d_prime = min(D, m-d);  
       for(i in 1:S){
        if(i != s){
         l_alpha_star[pointer] = log_sum_exp(to_vector(l_alpha[m-d, i, 1:d_prime]) + to_vector(l_tpm[i, s, m-d, 1:d_prime]));
         pointer += 1;
        }
       } 
       l_alpha[m, s, d] = log_sum_exp(l_alpha_star) + log_density_duration[s, m-d+1, d] + time_log_lik[s, m-d+1, m] + obs_log_lik[s, m-d+1, m];
      }    
     }
    }
    
    //Integral over all durations
    for(s in 1:S){
     log_lik[s] = log_sum_exp(l_alpha[M, s]);
    }
        
    return log_sum_exp(log_lik); //log P(y_{1:T}|\lambda) - loglikelihood      
}
}
data {
  int<lower=1> S; // number of transient states
  int<lower=1> P; // number of possible pages
  int<lower=1> D; // max state duration allowed
  int<lower=1> N_train; // number of users in training set
  int<lower=1> max_length;
  int<lower=1> M_train[N_train]; // pages per sequence
  int<lower=1, upper=2> L_train[N_train]; // outcome - 1 is exit & 2 is purchase
  real Y_train[N_train, max_length];// time on page
  int obs_train[N_train, max_length+1];
}
parameters {
  simplex[S] pi ; //initial state distribution for each user
  matrix[S, S] alpha_trans; //transition intercept
  vector[S] alpha; //gamma shape parameter
  vector[S] HP_ratio;// Hawkes process alpha
  vector[S] HP_beta;// Hawkes process beta
  vector[S] HP_base_intensity;// Hawkes process base intensity
  matrix[S, P] phi_0;//MAR intercept for each page in each state
  real phi_1[S, P, P];//MAR intercept for each page in each state
}
transformed parameters{
    vector[N_train] log_lik;
    vector[S] l_pi=log(pi);
    
    for(n in 1:N_train){
        int CP = M_train[n];
        matrix[CP-1, CP-1] duration_matrix;
        real log_density_duration[S, CP-1, CP-1] = rep_array(negative_infinity(), S, CP-1, CP-1);//state duration density for all possible sojourn time and states
        vector[CP] Time;
        real time_log_lik[S, CP-1, CP-1] = rep_array(negative_infinity(), S, CP-1, CP-1);
        real obs_log_lik[S, CP, CP] = rep_array(negative_infinity(), S, CP, CP);
        real l_tpm[S, S, CP-1, D];
        real acc[S-1];
        int pointer;
        //Observation times
        Time[1] = 0;
        Time[2:(CP)] = to_vector(cumulative_sum(Y_train[n, 1:(CP-1)]));
      
        //likelihood of possible combinations of observation times
        for(s in 1:(S)){
            for(i in 1:(CP-1)){
                for(j in i:(CP-1)){
                    time_log_lik[s, i, j] = compute_log_lik_HP(Time[i:j] - Time[i], j-i+1, Time[j+1] - Time[i], HP_beta[s], HP_base_intensity[s], HP_ratio[s]);
                }
            }
        } 
       
        //p(y|t)
        for(s in 1:(S)){
            for(i in 1:CP){
                for(j in i:CP){
                    obs_log_lik[s, i, j] = compute_log_lik_obs(obs_train[n, i:j], Y_train[n, i:(j-1)], phi_0[s], phi_1[s]);
                }
            }
        }
       
        //Duration matrix
        for(i in 1:(CP-1)){
            duration_matrix[i,] = rep_row_vector(0.0, (CP-1));
            duration_matrix[i, 1:((CP-1)-i+1)] = to_row_vector(cumulative_sum(Y_train[n, i:(CP-1)]));
        }
    
        //density_duration
        for(s in 1:S){
            for(i in 1:(CP-1)){
                for(j in 1:(CP-1)){
                    if(j <= ((CP-1)-i+1)){
                        log_density_duration[s, i, j] = log(alpha[s]) - alpha[s] * duration_matrix[i, j];//alpha[s] * log(beta[s]) - log(tgamma(alpha[s]))+(alpha[s]-1)*log(duration_matrix[i, j])-beta[s]*duration_matrix[i, j];//gamma_lpdf(duration_matrix[i, j] | alpha[s], beta[s]);
                    }
                }
            }    
        }
    
        //Transition matrix
        for(s_from in 1:S){   
            for(m in 1:(CP-1)){
                for(d in 1:D){
                    pointer = 1;
                    acc = rep_array(0.0, S-1);
                    for(s in 1:(S)){
                        if(d <= min(D, m)){
                            l_tpm[s_from, s, m, d] = transition_prob(s_from, s, m, d, rep_matrix(0.0, S, S), alpha_trans, Y_train[n]);
                            if(s != s_from){
                                acc[pointer] = l_tpm[s_from, s, m, d];
                                pointer += 1;
                            }       
                        }
                        else{
                            l_tpm[s_from, s, m, d] = negative_infinity();
                        }
                    } 
                    //Scale so [0,1]
                    for(s in 1:(S)){
                        if(s != s_from){
                            l_tpm[s_from, s, m, d] -= log_sum_exp(acc);
                        }
                    }    
                } 
            }
        }      
        log_lik[n] = compute_log_lik(Time , CP-1, D , S , L_train[n] , l_pi , log_density_duration, time_log_lik, obs_log_lik, l_tpm);
    }
}
model {
    pi ~ dirichlet(rep_vector(1,S));
    alpha ~ gamma(2, 0.1);
    HP_ratio ~ gamma(2, 0.1);
    HP_beta ~ gamma(2, 0.1);
    HP_base_intensity ~ gamma(2, 0.1);
    for(s in 1:S){
      alpha_trans[s] ~ normal(0, 5);
    }
    target += sum(log_lik);
}
